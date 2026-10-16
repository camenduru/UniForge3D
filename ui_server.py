"""UniForge3D web UI: upload views, tune settings, run, inspect the GLB.

Run:  .venv/bin/python ui_server.py [--port 8091]
"""
import argparse
import base64
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
import sys as _sys
_VENV_PY = ROOT / ".venv" / "bin" / "python"
VENV_PY = str(_VENV_PY if _VENV_PY.exists() else Path(_sys.executable))
RUNNER = str(ROOT / "ui_run.py")
INPUTS_DIR = ROOT / "ui_inputs"
OUTPUTS_DIR = ROOT / "outputs"
JOBS_DIR = Path("/tmp/opencode/ui_jobs")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def base_env() -> dict:
    env = dict(os.environ)
    env["CUDA_HOME"] = "/usr/local/cuda-12.8"
    env["PATH"] = "/usr/local/cuda-12.8/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + env.get("LD_LIBRARY_PATH", "")
    env["USE_TORCH"] = "1"
    # Reduce CUDA fragmentation OOMs (PyTorch's own suggestion on OOM).
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # A user-shell PYTHONPATH pointing at system dist-packages would shadow
    # the venv's numpy-2-compatible builds (shapely, sklearn, ...). Drop it.
    env.pop("PYTHONPATH", None)
    return env


MIN_FREE_VRAM_MB = 18000
MAX_FOREIGN_MB = 500  # any other process holding more than this blocks a run


def vram_status() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        used, free, total = (int(float(x)) for x in out.stdout.strip().split(","))
        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception as exc:
        return {"error": str(exc)[:200]}


def gpu_holders() -> list[dict]:
    """Other processes currently holding VRAM (pid, name, MB)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        holders = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            pid, name, mb = int(parts[0]), parts[1], int(float(parts[2]))
            if pid != os.getpid() and mb > 0:
                holders.append({"pid": pid, "name": name.split("/")[-1][:40],
                                "used_mb": mb})
        return holders
    except Exception:
        return []


def preflight() -> str | None:
    """Return an error message if a run must not start, else None."""
    busy = active_job()
    if busy is not None:
        return (f"GPU is busy with '{busy['case']}' (job {busy['job']}). "
                "Wait for it to finish; only one generation runs at a time.")
    holders = [h for h in gpu_holders() if h["used_mb"] > MAX_FOREIGN_MB]
    if holders:
        desc = ", ".join(f"{h['name']} (pid {h['pid']}, "
                         f"{h['used_mb'] / 1024:.1f} GB)" for h in holders)
        return (f"GPU is held by another process: {desc}. Stop it first — "
                "sharing the card mid-run causes out-of-memory failures.")
    vram = vram_status()
    if "free_mb" in vram and vram["free_mb"] < MIN_FREE_VRAM_MB:
        return (f"Only {vram['free_mb'] / 1024:.1f} GB VRAM free; "
                f"{MIN_FREE_VRAM_MB / 1024:.0f} GB required. Free the GPU and retry.")
    return None


def active_job() -> dict | None:
    with JOBS_LOCK:
        for job_id, job in JOBS.items():
            if job["proc"].poll() is None:
                return {"job": job_id, **{k: v for k, v in job.items() if k != "proc"}}
    return None


def kill_tree(proc: subprocess.Popen, timeout: float = 12.0) -> None:
    """Stop a generation and make sure nothing of it is left running.

    The runner is launched with start_new_session=True, so its pid is also its
    own process-group id: signalling the group can never reach this server.
    SIGTERM first (so the runner can unwind), then SIGKILL. Killing the process
    is what actually frees the VRAM — the CUDA context is torn down with it.
    """
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    if pgid == os.getpgrp():        # belt and braces: never signal our own group
        pgid = None

    def send(sig: int) -> None:
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    for sig in (signal.SIGTERM, signal.SIGKILL):
        send(sig)
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def wait_for_vram_release(timeout: float = 15.0, want_mb: int = MIN_FREE_VRAM_MB) -> dict:
    """Wait until the driver has actually handed the killed run's VRAM back.

    The CUDA context unwinds asynchronously after the process exits, so poll
    rather than assume. Stops early once enough is free, or once the figure has
    held steady for 3 s (another process may legitimately own the rest).
    """
    deadline = time.time() + timeout
    period = 0.5
    vram = vram_status()
    last = vram.get("free_mb")
    unchanged = 0.0
    while time.time() < deadline:
        if (last or 0) >= want_mb:
            break
        time.sleep(period)
        vram = vram_status()
        free = vram.get("free_mb")
        unchanged = unchanged + period if free == last else 0.0
        last = free
        if unchanged >= 3.0:
            break
    return vram


def safe_case(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (name or "").strip())
    return keep[:64] or f"case_{time.strftime('%Y%m%d_%H%M%S')}"


def list_outputs() -> list[dict]:
    items = []
    if OUTPUTS_DIR.is_dir():
        for case in sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir()):
            for glb in sorted(case.glob("*.glb")):
                items.append({
                    "case": case.name,
                    "file": glb.name,
                    "url": f"/outputs/{case.name}/{glb.name}",
                    "size_mb": round(glb.stat().st_size / 1e6, 1),
                    "mtime": glb.stat().st_mtime,
                })
    return items


def find_gltfpack() -> str | None:
    import shutil
    for cand in ("/usr/local/bin/gltfpack",
                 str(ROOT / "tools" / "gltfpack"),
                 shutil.which("gltfpack") or ""):
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def list_inputs(case: str) -> list[dict]:
    d = INPUTS_DIR / case
    if not d.is_dir():
        return []
    return [{"name": p.name, "url": f"/inputs/{case}/{p.name}"}
            for p in sorted(d.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]


class Handler(BaseHTTPRequestHandler):
    server_version = "UniForgeUI/1.0"

    def log_message(self, *args):
        pass

    def _json(self, obj: dict, code: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def _file(self, path: Path, ctype: str):
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Browsers percent-encode spaces/unicode in filenames; decode before
        # matching so "Screenshot 2026-10-16 110451.jpg" resolves.
        parsed = urlparse(unquote(self.path))
        if parsed.path == "/":
            return self._html()
        if parsed.path == "/api/outputs":
            return self._json({"outputs": list_outputs()})
        if parsed.path == "/api/gpu":
            return self._json({"vram": vram_status(), "active": active_job(),
                               "holders": gpu_holders()})
        if parsed.path == "/api/inputs":
            from urllib.parse import parse_qs
            case = safe_case(parse_qs(parsed.query).get("case", [""])[0])
            return self._json({"case": case, "files": list_inputs(case)})
        if parsed.path == "/api/log":
            from urllib.parse import parse_qs
            q = parse_qs(parsed.query)
            job_id = q.get("job", [""])[0]
            offset = int(q.get("offset", ["0"])[0] or 0)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                return self._json({"error": "unknown job"}, 404)
            log = JOBS_DIR / f"{job_id}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            running = job["proc"].poll() is None
            return self._json({
                "running": running,
                "canceled": bool(job.get("canceled", False)),
                "returncode": job["proc"].poll(),
                "elapsed": round(time.time() - job["started"], 1),
                "log": text[offset:],
                "next_offset": len(text),
            })
        if parsed.path.startswith("/outputs/"):
            rel = Path(parsed.path[len("/outputs/"):])
            if ".." in rel.parts:
                return self.send_error(403)
            return self._file(OUTPUTS_DIR / rel, "model/gltf-binary")
        if parsed.path.startswith("/inputs/"):
            rel = Path(parsed.path[len("/inputs/"):])
            if ".." in rel.parts:
                return self.send_error(403)
            p = INPUTS_DIR / rel
            ctype = {".png": "image/png", ".jpg": "image/jpeg",
                     ".jpeg": "image/jpeg", ".webp": "image/webp",
                     ".bmp": "image/bmp"}.get(p.suffix.lower(), "application/octet-stream")
            return self._file(p, ctype)
        return self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON"}, 400)

        if self.path == "/api/upload":
            case = safe_case(payload.get("case", ""))
            files = payload.get("files", [])
            if not files:
                return self._json({"error": "no files"}, 400)
            case_dir = INPUTS_DIR / case
            case_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for f in files:
                name = Path(f.get("name", "img.png")).name
                suffix = Path(name).suffix.lower()
                if suffix not in IMAGE_SUFFIXES:
                    continue
                (case_dir / name).write_bytes(base64.b64decode(f["data"]))
                saved.append(name)
            return self._json({"case": case, "saved": saved,
                               "files": list_inputs(case)})

        if self.path == "/api/run":
            case = safe_case(payload.get("case", ""))
            err = preflight()
            if err is not None:
                return self._json({"error": err}, 409)
            images = [INPUTS_DIR / case / f["name"] for f in list_inputs(case)]
            if not images:
                return self._json({"error": f"no images uploaded for '{case}'"}, 400)
            model = "xibushu/UniForge3D"
            seed = int(payload.get("seed", 42))
            low_vram = bool(payload.get("low_vram", True))
            overwrite = bool(payload.get("overwrite", True))
            device = int(payload.get("device", 0))
            skip_rembg = bool(payload.get("skip_rembg", False))
            rembg_model = "camenduru/RMBG-2.0"
            out = OUTPUTS_DIR / case / "prediction.glb"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [VENV_PY, RUNNER,
                   "--images", *[str(p) for p in images],
                   "--output", str(out), "--model", model,
                   "--seed", str(seed), "--device", str(device)]
            cmd += ["--low-vram" if low_vram else "--no-low-vram"]
            if overwrite:
                cmd.append("--overwrite")
            if skip_rembg:
                cmd.append("--skip-rembg")
            else:
                cmd += ["--rembg-model", rembg_model]
            job_id = uuid.uuid4().hex[:8]
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            log = open(JOBS_DIR / f"{job_id}.log", "w")
            log.write(f"$ {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=base_env(), cwd=str(ROOT),
                                    start_new_session=True)
            with JOBS_LOCK:
                JOBS[job_id] = {"proc": proc, "started": time.time(),
                                "case": case, "output": str(out),
                                "canceled": False}
            return self._json({"job": job_id, "case": case,
                               "output_url": f"/outputs/{case}/prediction.glb"})

        if self.path == "/api/cancel":
            # Stop the active generation, then wait for its VRAM to come back so
            # a new run can start immediately (preflight needs ~18 GB free).
            job_id = str(payload.get("job", "")).strip()
            with JOBS_LOCK:
                job = JOBS.get(job_id) if job_id else None
                if job is None or job["proc"].poll() is not None:
                    job = next((j for j in JOBS.values()
                                if j["proc"].poll() is None), None)
                proc = job["proc"] if job is not None else None
                if job is not None:
                    job["canceled"] = True
            if proc is None:
                return self._json({"error": "no generation is running"}, 409)
            kill_tree(proc)
            vram = wait_for_vram_release()
            return self._json({
                "canceled": True,
                "returncode": proc.poll(),
                "vram": vram,
                "holders": gpu_holders(),
                "free_mb": vram.get("free_mb"),
            })

        if self.path == "/api/delete":
            case = safe_case(payload.get("case", ""))
            name = Path(payload.get("name", "")).name
            target = INPUTS_DIR / case / name
            if target.is_file() and target.suffix.lower() in IMAGE_SUFFIXES:
                target.unlink()
            return self._json({"case": case, "files": list_inputs(case)})

        if self.path == "/api/optimize":
            case = safe_case(payload.get("case", ""))
            source = Path(payload.get("source", "prediction.glb")).name
            src = OUTPUTS_DIR / case / source
            if not src.is_file() or src.suffix.lower() != ".glb":
                return self._json({"error": "source .glb not found"}, 400)
            gltfpack = find_gltfpack()
            if gltfpack is None:
                return self._json({"error": "gltfpack binary not installed"}, 500)
            try:
                ratio = float(payload.get("ratio", 0.25))
                error = float(payload.get("error", 0.01))
            except (TypeError, ValueError):
                return self._json({"error": "bad ratio/error"}, 400)
            ratio = min(1.0, max(0.01, ratio))
            error = min(1.0, max(0.0, error))
            out_name = Path(payload.get("output", "optimized.glb")).name
            if not out_name.endswith(".glb"):
                out_name += ".glb"
            out = OUTPUTS_DIR / case / out_name
            cmd = [gltfpack, "-i", str(src), "-o", str(out),
                   "-si", str(ratio), "-se", str(error)]
            if payload.get("aggressive"):
                cmd.append("-sa")
            compress = payload.get("compress", "meshopt")
            if compress == "meshopt":
                cmd.append("-cc")
            elif compress == "quantized":
                cmd.append("-c")
            if payload.get("webp"):
                cmd += ["-tw"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0 or not out.is_file():
                return self._json({"error": (proc.stderr or proc.stdout)[-500:]}, 500)
            return self._json({
                "case": case, "file": out.name,
                "url": f"/outputs/{case}/{out.name}",
                "size_mb": round(out.stat().st_size / 1e6, 1),
                "src_mb": round(src.stat().st_size / 1e6, 1),
            })

        return self._json({"error": "unknown endpoint"}, 404)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>UniForge3D Studio</title>
<script src="https://cdn.babylonjs.com/babylon.js"></script>
<script src="https://cdn.babylonjs.com/loaders/babylonjs.loaders.min.js"></script>
<style>
:root { --bg:#0f1115; --panel:#171b22; --line:#262c38; --txt:#e8ecf3; --mut:#9aa3b5;
  --acc:#4f8cff; --ok:#3ecf8e; --bad:#ff6b6b; }
* { box-sizing: border-box; }
html,body { margin:0; min-height:100%; background:var(--bg); color:var(--txt);
  font:14px/1.5 system-ui,sans-serif; }
header { padding:14px 20px; border-bottom:1px solid var(--line); background:#13161c;
  position:sticky; top:0; z-index:5; }
header h1 { margin:0; font-size:18px; }
header p { margin:2px 0 0; color:var(--mut); font-size:12px; }
main { display:grid; grid-template-columns:380px 1fr; gap:16px; padding:16px;
  max-width:1500px; margin:0 auto; align-items:start; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin-bottom:16px; }
.card h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--mut); }
label { display:block; font-size:12px; color:var(--mut); margin:10px 0 4px; }
input[type=text],input[type=number],select { width:100%; padding:8px 10px;
  background:#0d1015; border:1px solid var(--line); color:var(--txt);
  border-radius:6px; font-size:13px; }
.row { display:flex; gap:10px; align-items:center; }
.row > * { flex:1; }
.check { display:flex; gap:8px; align-items:center; margin:10px 0; font-size:13px; }
.check input { width:16px; height:16px; accent-color:var(--acc); }
button { cursor:pointer; border:0; border-radius:6px; font-size:13px; font-weight:600; }
.btn-run { width:100%; padding:12px; background:var(--acc); color:#fff; font-size:15px;
  margin-top:12px; }
.btn-run:disabled { opacity:.5; cursor:wait; }
.btn-ghost { background:#232936; color:var(--txt); padding:6px 12px; }
.hint { font-size:12px; color:var(--mut); margin-top:8px; }
#thumbs { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin-top:8px; }
.thumb { position:relative; }
.thumb img { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px;
  border:1px solid var(--line); display:block; }
.thumb button { position:absolute; top:4px; right:4px; width:24px; height:24px;
  border-radius:50%; background:rgba(0,0,0,.7); color:#fff; font-size:14px;
  line-height:1; padding:0; }
#status { font-size:13px; padding:8px 10px; border-radius:6px; margin-top:10px;
  background:#0d1015; border:1px solid var(--line); }
#status.running { border-color:var(--acc); }
#status.done { border-color:var(--ok); }
#status.failed { border-color:var(--bad); }
#status.canceled { border-color:#d8a657; }
#log { height:260px; overflow-y:auto; background:#0a0c10; border:1px solid var(--line);
  border-radius:6px; padding:10px; font:11px/1.5 ui-monospace,monospace;
  white-space:pre-wrap; word-break:break-all; }
#glcanvas { width:100%; height:560px; background:#1c212b; border-radius:6px;
  border:1px solid var(--line); display:block; outline:none;
  touch-action:none; overscroll-behavior:contain; }
#gallery { display:flex; flex-wrap:wrap; gap:8px; }
#gallery button.on { outline:2px solid var(--acc); }
footer { color:var(--mut); font-size:12px; text-align:center; padding:12px; }
</style>
</head>
<body>
<header>
  <h1>UniForge3D Studio</h1>
  <p>Sparse unposed multi-view images &rarr; textured 3D asset (.glb)</p>
</header>
<main>
<div>
  <div class="card">
    <h2>1 &middot; Source views</h2>
    <label>Case name</label>
    <input id="case" type="text" value="case01" />
    <label>Images (2+ unposed views, PNG/JPG/WebP)</label>
    <input id="files" type="file" accept="image/*" multiple />
    <div id="thumbs"></div>
    <div class="hint" id="upstat">No images loaded.</div>
  </div>
  <div class="card">
    <h2>2 &middot; Settings</h2>
    <div class="row">
      <div><label>Seed</label><input id="seed" type="number" value="42" /></div>
      <div><label>GPU device</label><input id="device" type="number" value="0" min="0" /></div>
    </div>
    <div class="check"><input id="lowvram" type="checkbox" checked />
      <span>Low-VRAM mode <small style="color:var(--mut)">(offload stages to CPU, fits 24GB)</small></span></div>
    <div class="check"><input id="overwrite" type="checkbox" checked />
      <span>Overwrite existing output</span></div>
    <div class="check"><input id="skiprembg" type="checkbox" />
      <span>Skip background removal <small style="color:var(--mut)">(only if input PNGs are already transparent)</small></span></div>
    <div class="hint">Opaque images (JPG / flattened PNG) are the default: background
      removal runs automatically via the baked-in public
      <code>camenduru/RMBG-2.0</code> mirror, no login needed. Tick "skip" only
      when every uploaded view already has a transparent alpha channel.</div>
    <button class="btn-run" id="run">Generate 3D</button>
    <button class="btn-ghost" id="cancel" disabled style="width:100%;margin-top:8px">Cancel generation</button>
    <div class="hint" id="gpu">GPU: checking...</div>
    <div id="status">Idle.</div>
    <div class="hint">Stages: DINO encode &rarr; sparse structure &rarr; shape-512 &rarr;
      shape-1024 &rarr; texture bake &rarr; .glb export. Takes ~10 min on RTX 3090.</div>
  </div>
  <div class="card">
    <h2>3 &middot; Past results</h2>
    <div id="gallery"></div>
  </div>
  <div class="card">
    <h2>4 &middot; Optimize for games <small style="font-weight:400">(meshoptimizer gltfpack)</small></h2>
    <label>Source file</label>
    <select id="optsrc"></select>
    <div class="row">
      <div><label>Triangle ratio (0.01–1)</label><input id="optratio" type="number" value="0.25" min="0.01" max="1" step="0.05" /></div>
      <div><label>Max error (0–1)</label><input id="opterr" type="number" value="0.01" min="0" max="1" step="0.005" /></div>
    </div>
    <div class="row">
      <div><label>Compression</label><select id="optcomp">
        <option value="meshopt" selected>meshopt (-cc, smallest)</option>
        <option value="quantized">quantized (-c)</option>
        <option value="none">none (simplify only)</option>
      </select></div>
      <div><label>Output name</label><input id="optout" type="text" value="optimized.glb" /></div>
    </div>
    <div class="check"><input id="optaggr" type="checkbox" />
      <span>Aggressive simplify <small style="color:var(--mut)">(hit ratio, lower quality)</small></span></div>
    <div class="check"><input id="optwebp" type="checkbox" />
      <span>Textures to WebP <small style="color:var(--mut)">(smaller, needs WebP viewers)</small></span></div>
    <button class="btn-run" id="optbtn" style="font-size:13px">Optimize mesh</button>
    <div class="hint" id="optstat">Typical: 1M tris / 45 MB &rarr; 250k tris / ~6 MB.</div>
  </div>
</div>
<div>
  <div class="card">
    <h2>Result viewer</h2>
    <canvas id="glcanvas"></canvas>
    <div class="hint" id="mvstat">Viewer ready.</div>
    <div class="row" style="margin-top:10px">
      <button class="btn-ghost" id="dl">Download .glb</button>
      <button class="btn-ghost" id="autorot">Toggle auto-rotate</button>
      <button class="btn-ghost" id="resetview">Reset view</button>
    </div>
  </div>
  <div class="card">
    <h2>Run log</h2>
    <div id="log"></div>
  </div>
</div>
</main>
<footer>UniForge3D &middot; joint reconstruction + generation in a shared anchor-guided space</footer>
<script>
const $ = id => document.getElementById(id);
let pollTimer = null, logOffset = 0, currentUrl = "", activeJob = null;

async function refreshInputs() {
  const c = $("case").value.trim() || "case01";
  const r = await fetch("/api/inputs?case=" + encodeURIComponent(c)).then(r => r.json());
  const t = $("thumbs"); t.innerHTML = "";
  r.files.forEach(f => {
    const wrap = document.createElement("div"); wrap.className = "thumb";
    const img = document.createElement("img"); img.src = f.url;
    const x = document.createElement("button"); x.textContent = "×"; x.title = "Remove " + f.name;
    x.onclick = async () => {
      await fetch("/api/delete", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({case: $("case").value, name: f.name})});
      refreshInputs();
    };
    wrap.appendChild(img); wrap.appendChild(x); t.appendChild(wrap);
  });
  $("upstat").textContent = r.files.length ? r.files.length + " image(s) in '" + r.case + "'." : "No images loaded.";
}
async function refreshGallery() {
  const r = await fetch("/api/outputs").then(r => r.json());
  const g = $("gallery"); g.innerHTML = "";
  const sel = $("optsrc"); sel.innerHTML = "";
  r.outputs.forEach(o => {
    const b = document.createElement("button");
    b.className = "btn-ghost" + (o.url === currentUrl ? " on" : "");
    b.textContent = o.case + "/" + o.file + " (" + o.size_mb + " MB)";
    b.onclick = () => showResult(o.url);
    g.appendChild(b);
    const opt = document.createElement("option");
    opt.value = JSON.stringify({case: o.case, file: o.file});
    opt.textContent = o.case + "/" + o.file;
    sel.appendChild(opt);
  });
  if (!r.outputs.length) g.innerHTML = "<span class='hint'>No outputs yet.</span>";
}
$("optbtn").addEventListener("click", async () => {
  const src = JSON.parse($("optsrc").value || "null");
  if (!src) { $("optstat").textContent = "No source file."; return; }
  $("optbtn").disabled = true;
  $("optstat").textContent = "Optimizing...";
  const body = {case: src.case, source: src.file,
    ratio: +$("optratio").value, error: +$("opterr").value,
    aggressive: $("optaggr").checked, compress: $("optcomp").value,
    webp: $("optwebp").checked, output: $("optout").value};
  const r = await fetch("/api/optimize", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)}).then(r => r.json());
  $("optbtn").disabled = false;
  if (r.error) { $("optstat").textContent = "Failed: " + r.error; return; }
  $("optstat").textContent = "Done: " + r.src_mb + " MB -> " + r.size_mb + " MB.";
  showResult(r.url);
});
let engine = null, scene = null, camera = null, wantSpin = true;
let pendingUrl = null, babylonRetries = 0;
function showResult(url) {
  currentUrl = url;
  if (!window.BABYLON || !BABYLON.SceneLoader) {
    // CDN scripts may still be loading (or blocked): retry briefly, then say so.
    if (babylonRetries < 15) {
      babylonRetries++;
      pendingUrl = url;
      setTimeout(() => { if (pendingUrl === url) showResult(url); }, 1000);
      $("mvstat").textContent = "Waiting for 3D libraries...";
    } else {
      pendingUrl = null;
      $("mvstat").textContent = "3D libraries did not load (CDN unreachable?). " +
        "Downloads below still work.";
    }
    return;
  }
  babylonRetries = 0; pendingUrl = null;
  if (!engine) {
    engine = new BABYLON.Engine($("glcanvas"), true);
    window.addEventListener("resize", () => engine.resize());
    engine.runRenderLoop(() => { if (scene) scene.render(); });
  }
  if (scene) scene.dispose();
  scene = new BABYLON.Scene(engine);
  scene.clearColor = new BABYLON.Color4(0.11, 0.13, 0.17, 1);
  camera = new BABYLON.ArcRotateCamera("cam", -Math.PI / 2, Math.PI / 2.5, 3,
    BABYLON.Vector3.Zero(), scene);
  // `false` here means "do preventDefault": without it the wheel event also
  // scrolls the page behind the canvas while you zoom.
  camera.attachControl($("glcanvas"), false);
  // Zoom is driven by our own wheel handler (below). Babylon's built-in wheel
  // zoom is asymmetric: it scales the zoom-OUT delta by (1 + wheelDeltaPercentage)
  // and the zoom-IN delta by 1 / (1 + p), and it only compensates inertia when
  // zooming IN — so scrolling out ran away while scrolling in felt fine.
  // Dropping its wheel input leaves rotation/panning on the pointer input and
  // two-finger pinch zoom on pinchDeltaPercentage.
  camera.inputs.removeByType("ArcRotateCameraMouseWheelInput");
  camera.pinchDeltaPercentage = 0.1;
  camera.pinchPrecision = 30;
  camera.useAutoRotationBehavior = wantSpin;
  camera.autoRotationBehavior.idleRotationSpeed = 0.5;
  new BABYLON.HemisphericLight("h", new BABYLON.Vector3(0, 1, 0), scene).intensity = 1.1;
  const sun = new BABYLON.DirectionalLight("sun", new BABYLON.Vector3(-1, -2, 1), scene);
  sun.intensity = 0.7;
  $("mvstat").textContent = "Loading " + url + " ...";
  const full = url + "?t=" + Date.now();
  const slash = full.lastIndexOf("/");
  BABYLON.SceneLoader.Append(full.slice(0, slash + 1), full.slice(slash + 1), scene,
    () => {
      const meshes = scene.meshes.filter(m => m.getTotalVertices() > 0);
      if (meshes.length) {
        let min = meshes[0].getBoundingInfo().boundingBox.minimumWorld.clone();
        let max = meshes[0].getBoundingInfo().boundingBox.maximumWorld.clone();
        meshes.forEach(m => {
          const b = m.getBoundingInfo().boundingBox;
          min.minimizeInPlace(b.minimumWorld); max.maximizeInPlace(b.maximumWorld);
        });
        camera.setTarget(min.add(max).scale(0.5));
        const diagonal = Math.max(max.subtract(min).length(), 0.01);
        camera.radius = diagonal * 1.1;
        // Keep zoom sensible: not clipping into the surface, not lost in space.
        camera.lowerRadiusLimit = diagonal * 0.2;
        camera.upperRadiusLimit = diagonal * 10;
        camera.inertialRadiusOffset = 0;
      }
      $("mvstat").textContent = "Loaded: " + currentUrl +
        " (" + scene.meshes.length + " node(s))";
      refreshGallery();
    },
    undefined,
    (s, msg) => {
      $("mvstat").textContent = "Viewer failed to load " + currentUrl +
        " (" + (msg || "load error") + "). Try Download to verify the file.";
    });
  refreshGallery();
}
setTimeout(() => {
  if (!window.BABYLON) {
    $("mvstat").textContent = "3D libraries did not load (CDN unreachable?). " +
      "Viewer needs https://cdn.babylonjs.com access; downloads below still work.";
  }
}, 8000);
$("files").addEventListener("change", async e => {
  const files = [...e.target.files];
  if (!files.length) return;
  $("upstat").textContent = "Uploading " + files.length + " file(s)...";
  const data = await Promise.all(files.map(f => new Promise(res => {
    const rd = new FileReader();
    rd.onload = () => res({name: f.name, data: rd.result.split(",")[1]});
    rd.readAsDataURL(f);
  })));
  const r = await fetch("/api/upload", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({case: $("case").value, files: data})}).then(r => r.json());
  if (r.error) { $("upstat").textContent = "Upload failed: " + r.error; return; }
  $("case").value = r.case;
  refreshInputs();
});
$("case").addEventListener("change", refreshInputs);
$("run").addEventListener("click", async () => {
  clearInterval(pollTimer); logOffset = 0; $("log").textContent = "";
  const body = {case: $("case").value,
    seed: +$("seed").value, device: +$("device").value,
    low_vram: $("lowvram").checked, overwrite: $("overwrite").checked,
    skip_rembg: $("skiprembg").checked};
  $("run").disabled = true;
  $("cancel").disabled = true;
  const st = $("status"); st.className = "running"; st.textContent = "Starting...";
  const r = await fetch("/api/run", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)}).then(r => r.json());
  if (r.error) { st.className = "failed"; st.textContent = "Error: " + r.error;
    $("run").disabled = false; return; }
  activeJob = r.job;
  $("cancel").disabled = false;
  const t0 = Date.now();
  pollTimer = setInterval(async () => {
    const s = await fetch("/api/log?job=" + r.job + "&offset=" + logOffset).then(r => r.json());
    if (s.log) { $("log").textContent += s.log; $("log").scrollTop = 1e9; logOffset = s.next_offset; }
    const el = ((Date.now() - t0) / 1000).toFixed(0) + "s elapsed";
    if (s.running) { st.textContent = "Generating '" + r.case + "'... " + el; }
    else {
      clearInterval(pollTimer); activeJob = null;
      $("run").disabled = false; $("cancel").disabled = true;
      if (s.canceled) { st.className = "canceled";
        st.textContent = "Cancelled after " + el + " — VRAM released.";
      } else if (s.returncode === 0) { st.className = "done";
        st.textContent = "Done in " + el + ". Output: outputs/" + r.case + "/prediction.glb";
        showResult(r.output_url);
      } else { st.className = "failed";
        st.textContent = "Failed (exit " + s.returncode + ") after " + el + ". See log."; }
    }
  }, 2000);
});
$("cancel").addEventListener("click", async () => {
  const c = $("cancel");
  c.disabled = true; c.textContent = "Cancelling...";
  const st = $("status"); st.className = "running";
  st.textContent = "Cancelling and releasing VRAM...";
  const r = await fetch("/api/cancel", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({job: activeJob || ""})}).then(r => r.json());
  c.textContent = "Cancel generation";
  refreshGpu();
  if (r.error) { st.className = "failed"; st.textContent = r.error;
    c.disabled = false; return; }
  const free = (r.vram && r.vram.free_mb !== undefined)
    ? (r.vram.free_mb / 1024).toFixed(1) + " GB VRAM free" : "VRAM released";
  // The log poll settles the final status; this covers a cancel between polls.
  if (!activeJob) { st.className = "canceled"; st.textContent = "Cancelled — " + free + "."; }
});
$("dl").onclick = () => { if (currentUrl) window.open(currentUrl, "_blank"); };
$("autorot").onclick = () => {
  wantSpin = !wantSpin;
  if (camera) camera.useAutoRotationBehavior = wantSpin;
};
$("resetview").onclick = () => { if (currentUrl) showResult(currentUrl); };
// One symmetric ratio per wheel notch, so zooming out and then back in returns
// you to exactly the same distance. Babylon's own handler is asymmetric and
// lets zoom-out glide on after you stop scrolling (see the camera setup).
const ZOOM_PER_NOTCH = 1.08;
$("glcanvas").addEventListener("wheel", evt => {
  if (!camera || !currentUrl) return;
  evt.preventDefault();          // never scroll the page behind the canvas
  // Normalise a notch: Chrome reports ~100px, Firefox ~3 lines (deltaMode 1).
  const scale = evt.deltaMode === 1 ? 40 : (evt.deltaMode === 2 ? 400 : 1);
  const steps = Math.max(-4, Math.min(4, (evt.deltaY * scale) / 100));
  const lo = camera.lowerRadiusLimit || 0.01;
  const hi = camera.upperRadiusLimit || Number.MAX_VALUE;
  camera.radius = Math.min(hi, Math.max(lo, camera.radius * Math.pow(ZOOM_PER_NOTCH, steps)));
  camera.inertialRadiusOffset = 0;   // zoom must not keep gliding after release
}, { passive: false });
async function refreshGpu() {
  try {
    const r = await fetch("/api/gpu").then(r => r.json());
    const v = r.vram || {};
    let txt = (v.free_mb !== undefined)
      ? "GPU VRAM free: " + (v.free_mb / 1024).toFixed(1) + " / " +
        (v.total_mb / 1024).toFixed(0) + " GB" : "GPU: n/a";
    if (r.active) txt += " — BUSY with '" + r.active.case + "'";
    if (r.holders && r.holders.length) txt += " — held by " +
      r.holders.map(h => h.name + " " + (h.used_mb / 1024).toFixed(1) + "GB").join(", ");
    $("gpu").textContent = txt;
  } catch (e) { $("gpu").textContent = "GPU: unreachable"; }
}
setInterval(refreshGpu, 5000); refreshGpu();
refreshInputs(); refreshGallery();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    INPUTS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"UniForge3D Studio on http://0.0.0.0:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
