

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Sequence

import torch
import trimesh
from huggingface_hub import snapshot_download

from ..models.loaders import load_models, resolve_checkpoint
from ..utils.device import release_cuda
from ..utils.mesh import (
    display_to_model,
    export_glb,
    normalize_to_cube,
    official_shape_only_postprocess,
    surface_voxelize,
)
from ..utils.preprocessing import BackgroundRemover, InputBatch, build_input_batch
from .shape import decode_shape_latent, sample_shape_latent
from .sparse_structure import decode_sparse_structure, decoded_coordinates, sample_sparse_structure
from .texture import (
    decode_shape,
    decode_texture,
    denormalize_shape_latent,
    remesh_and_bake_texture,
    sample_texture,
)
from .types import NormalizedShapeLatent, SparseStructure


class UniForge3DImageTo3DPipeline:


    def __init__(
        self,
        config: dict,
        *,
        device: int = 0,
        low_vram: bool = True,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.config = config
        self.device = torch.device("cuda", int(device))
        self.low_vram = low_vram

        with torch.device("cpu"):
            models = load_models(
                self.config, cache_dir=cache_dir, local_files_only=local_files_only,
            )
            self.dino = models["dino"]
            self.conditioner = models["conditioner"]
            self.structure_model = models["structure_model"]
            self.structure_decoder = models["structure_decoder"]
            self.shape512_model = models["shape512_model"]
            self.shape512_decoder = models["shape512_decoder"]
            self.shape1024_model = models["shape1024_model"]
            self.shape1024_decoder = models["shape1024_decoder"]
            self.texture_model = models["texture_model"]
            self.texture_decoder = models["texture_decoder"]
            self.remove_background = BackgroundRemover(
                self.config["image_preprocess"]["rembg_model"], self.device,
                cache_dir=cache_dir, local_files_only=local_files_only,
            )

    @classmethod
    def from_pretrained(
        cls, path: str | Path, *, device: int = 0, low_vram: bool = True,
        revision: str | None = None, cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a local model directory or a Hugging Face repository."""
        root = Path(path).expanduser()
        if not root.is_dir():
            root = Path(snapshot_download(
                str(path), revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=["pipeline.json", "uniforge3d/**/*.pt"],
            ))
        return cls.from_config(
            root / "pipeline.json", weights_root=root, device=device,
            low_vram=low_vram, cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    @classmethod
    def from_config(
        cls, config_path: str | Path, *, device: int = 0, low_vram: bool = True,
        weights_root: str | Path | None = None,
        cache_dir: str | Path | None = None, local_files_only: bool = False,
    ) -> "UniForge3DImageTo3DPipeline":
        """Load a JSON config; local weights default to the repository's weights/."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        root = (Path(weights_root).expanduser() if weights_root is not None
                else Path(__file__).resolve().parents[2] / "weights")
        options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        for stage in ("sparse_structure", "structured_latent_512", "structured_latent_1024", "texture"):
            weights = config[stage]["weights"]
            for key in ("checkpoint", "adapter_checkpoint", "dit_base", "vggt_base", "base_checkpoint"):
                if key in weights:
                    weights[key] = resolve_checkpoint(weights[key], root, **options)
            if "decoder" in config[stage]:
                decoder = config[stage]["decoder"]
                decoder["checkpoint"] = resolve_checkpoint(
                    decoder["checkpoint"], root, decoder=True, **options,
                )
        for key in ("shape_decoder_checkpoint", "texture_decoder_checkpoint"):
            config["texture"][key] = resolve_checkpoint(
                config["texture"][key], root, decoder=True, **options,
            )
        vggt = config["conditioner"]["vggt_feature_extractor"]
        vggt["checkpoint_path"] = resolve_checkpoint(vggt["checkpoint_path"], root, **options)
        return cls(config, device=device, low_vram=low_vram, **options)

    @contextmanager
    def _use_models(self, *models):

        try:
            for model in models:
                model.to(self.device)
            yield
        finally:
            if self.low_vram:
                for model in models:
                    model.to("cpu")
                release_cuda()

    def _preprocess_images(self, images: Sequence[str | Path]) -> InputBatch:

        with ExitStack() as stack:
            background_loaded = False

            def remove_background(image):
                nonlocal background_loaded
                if not background_loaded:
                    stack.enter_context(self._use_models(self.remove_background.model))
                    background_loaded = True
                return self.remove_background(image)

            return build_input_batch(
                images,
                self.config["image_preprocess"],
                remove_background,
            )

    @torch.no_grad()
    def _encode_dino(self, batch: InputBatch) -> dict[str, torch.Tensor]:
        """Encode once per resolution and share 512px features across stages."""
        with self._use_models(self.dino):
            images512 = batch.dino_images512.to(self.device, dtype=torch.float32)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                structure_features = self.dino(images512).cpu()
                shape1024_features = self.dino(
                    batch.dino_images1024.to(self.device, dtype=torch.float32)
                ).cpu()
        return {
            "sparse_structure": structure_features,
            "shape512": structure_features,
            "shape1024": shape1024_features,
        }

    def _generate_structure(
        self,
        batch: InputBatch,
        dino_features: torch.Tensor,
    ) -> SparseStructure:
        sampling = self.config["sparse_structure"]["sampler"]
        dit = self.structure_model.dit
        resolution = dit.resolution
        noise = torch.randn(
            (1, dit.in_channels, resolution, resolution, resolution),
            device="cpu",
        ).to(self.device)
        with self._use_models(self.structure_model):
            vggt_images = batch.vggt_images.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                latent = sample_sparse_structure(
                    self.structure_model,
                    noise,
                    vggt_images,
                    dino_features.to(self.device),
                    **sampling["args"],
                    **sampling["params"],
                )
        with self._use_models(self.structure_decoder):
            occupancy_logits = decode_sparse_structure(
                self.structure_decoder,
                latent,
                self.config["sparse_structure"]["decoder"].get("normalization"),
            )
        coords32, coords64 = decoded_coordinates(occupancy_logits)
        return SparseStructure(coords32.cpu(), coords64.cpu())

    def _sample_shape(
        self,
        model: torch.nn.Module,
        config: dict,
        batch: InputBatch,
        *,
        latent_coords: torch.Tensor,
        structure_conditioned_coords: torch.Tensor,
        dino_features: torch.Tensor,
        return_view_assignment: bool,
    ):
        latent_coords = latent_coords.int().contiguous()
        structure_conditioned_coords = structure_conditioned_coords.int().contiguous()
        with self._use_models(model, self.conditioner):
            conditioning = config["conditioning"]
            latent_coords = latent_coords.to(device=self.device, dtype=torch.int32)
            condition = self.conditioner.encode(
                model.adapter,
                dino_features=dino_features.to(device=self.device, dtype=torch.float32),
                vggt_images=batch.vggt_images.to(device=self.device, dtype=torch.float32),
                structure_conditioned_coords=structure_conditioned_coords,
                input_masks=batch.input_masks.to(device=self.device, dtype=torch.bool),
                latent_coords=latent_coords,
                router_grid=int(conditioning["grid_resolution"]),
            )
            latent = sample_shape_latent(
                model,
                latent_coords,
                condition,
                config["sampler"],
                description=f"Shape-{config['decoder']['resolution']} sampling",
            )
            view_assignment = (
                {"selected_view": condition["cond"]["selected_view"].cpu()}
                if return_view_assignment else None
            )
        if not torch.isfinite(latent.feats).all():
            raise RuntimeError("Shape sampling produced non-finite latent features")
        return latent, view_assignment

    def _generate_shape512(
        self,
        batch: InputBatch,
        structure: SparseStructure,
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        latent, _ = self._sample_shape(
            self.shape512_model,
            self.config["structured_latent_512"],
            batch,
            latent_coords=structure.coords32,
            structure_conditioned_coords=structure.coords64,
            dino_features=dino_features,
            return_view_assignment=False,
        )
        with self._use_models(self.shape512_decoder):
            mesh = decode_shape_latent(
                self.shape512_decoder,
                latent,
                self.config["structured_latent_512"]["decoder"].get("normalization"),
            )[0]
        return official_shape_only_postprocess(
            mesh.vertices,
            mesh.faces,
            **self.config["mesh_postprocess"],
        )

    def _generate_shape1024(
        self,
        batch: InputBatch,
        coords64: torch.Tensor,
        dino_features: torch.Tensor,
    ) -> tuple[NormalizedShapeLatent, dict[str, torch.Tensor]]:
        latent, view_assignment = self._sample_shape(
            self.shape1024_model,
            self.config["structured_latent_1024"],
            batch,
            latent_coords=coords64,
            structure_conditioned_coords=coords64,
            dino_features=dino_features,
            return_view_assignment=True,
        )
        if view_assignment is None:
            raise RuntimeError("Shape1024 must provide its texture routing result")
        shape = NormalizedShapeLatent(coords=latent.coords.cpu(), feats=latent.feats.cpu())
        return shape, view_assignment

    def _shape512_to_coords64(self, mesh: trimesh.Trimesh) -> torch.Tensor:
        """Convert the already-remeshed Shape-512 surface to coords64."""
        config = self.config["structured_latent_handoff"]
        model_vertices = display_to_model(mesh.vertices)
        normalized_vertices, _ = normalize_to_cube(model_vertices, config["target_max_side"])
        xyz = torch.from_numpy(
            surface_voxelize(normalized_vertices, mesh.faces, config["surface_voxel_resolution"])
        )
        return torch.cat((torch.zeros((len(xyz), 1), dtype=torch.int32), xyz.int()), dim=1)

    def _generate_texture(
        self,
        batch: InputBatch,
        normalized_shape_latent: NormalizedShapeLatent,
        view_assignment: dict[str, torch.Tensor],
        dino_features: torch.Tensor,
    ) -> trimesh.Trimesh:
        config = self.config["texture"]
        # Stage the models, never all at once. The 2048-channel texture flow
        # model (~5.5 GB) and the VGGT conditioner (~3.9 GB) are by far the
        # largest residents; holding them while the Shape-1024 decode runs
        # pushes a 24 GB card over the edge. So: sample the PBR latent, release
        # the flow model + conditioner, then decode shape, then decode texture.
        shape_vae_latent = denormalize_shape_latent(
            config,
            normalized_shape_latent,
            self.device,
        )
        with self._use_models(self.texture_model, self.conditioner):
            texture_vae_latent = sample_texture(
                self.texture_model,
                config,
                self.conditioner,
                shape_vae_latent,
                batch,
                dino_features,
                view_assignment,
                self.device,
            )
        with self._use_models(self.shape1024_decoder):
            reference_mesh, subdivisions = decode_shape(
                self.shape1024_decoder, shape_vae_latent
            )
        with self._use_models(self.texture_decoder):
            pbr_voxel = decode_texture(
                self.texture_decoder, texture_vae_latent, subdivisions
            )
        del shape_vae_latent, texture_vae_latent, subdivisions
        # Models are back on CPU and the CUDA cache was emptied on context
        # exit, so the peaky CuMesh hole filling gets the whole card.
        try:
            reference_mesh.fill_holes()
        except torch.cuda.OutOfMemoryError:
            release_cuda()
            try:
                reference_mesh.fill_holes()
            except torch.cuda.OutOfMemoryError as exc:
                from ..utils.device import vram_diagnostics
                raise RuntimeError(
                    "Out of GPU memory during mesh hole filling. Close other "
                    "GPU programs and run one generation at a time. "
                    + vram_diagnostics()
                ) from exc
        return remesh_and_bake_texture(
            config,
            self.config["mesh_postprocess"],
            reference_mesh,
            pbr_voxel,
        )

    @torch.no_grad()
    def run(
        self,
        images: Sequence[str | Path],
        output: str | Path,
        *,
        seed: int | None = None,
        overwrite: bool = False,
    ) -> Path:

        output_path = Path(output).expanduser().resolve()
        if output_path.exists() and (not overwrite or not output_path.is_file()):
            raise FileExistsError(
                f"Output already exists: {output_path}; use --overwrite to recompute"
            )
        seed = self.config["default_seed"] if seed is None else seed

        batch = self._preprocess_images(images)

        torch.cuda.set_device(self.device)
        torch.manual_seed(seed)
        try:
            # Share one DINO model while preserving each stage's original precision.
            dino_features = self._encode_dino(batch)

            sparse_structure = self._generate_structure(batch, dino_features["sparse_structure"])
            shape512_mesh = self._generate_shape512(
                batch, sparse_structure, dino_features["shape512"]
            )


            structure_conditioned_coords = self._shape512_to_coords64(shape512_mesh)
            shape1024_latent, view_assignment = self._generate_shape1024(
                batch,
                structure_conditioned_coords,
                dino_features["shape1024"],
            )

            # Texture reuses the Shape1024 latent and its per-token view routing.
            textured_mesh = self._generate_texture(
                batch,
                shape1024_latent,
                view_assignment,
                dino_features["shape1024"],
            )
            return export_glb(textured_mesh, output_path, overwrite=overwrite)
        finally:
            release_cuda()
