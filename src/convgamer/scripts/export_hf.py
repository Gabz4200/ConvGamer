"""Export ConvGamer model to HuggingFace format.

CLI: python -m convgamer.scripts.export_hf --checkpoint <path> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

_BACKBONE_KEYS = (
    "hidden_dim",
    "num_layers",
    "num_classes",
    "input_dim",
    "layer_scale_init",
    "mlp_ratios",
    "widths",
    "out_factor",
    "temporal_dilations",
    "patch_size",
    "correction_kernel",
    "expansion",
    "level_dim",
)


def _normalize_cfg(model_cfg: dict) -> dict:
    for key in ("num_layers", "mlp_ratios", "widths", "temporal_dilations"):
        if isinstance(model_cfg.get(key), list):
            model_cfg[key] = tuple(model_cfg[key])
    return model_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exported_model")
    parser.add_argument(
        "--ckpt-kind",
        type=str,
        choices=("video", "image", "jepa"),
        default="video",
        help="video/image load a BackboneClassifier; jepa loads a JEPA target encoder.",
    )
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint missing: {args.checkpoint}")

    from convgamer import __version__
    from convgamer.integrations.transformers import (
        ConvGamerImageConfig,
        ConvGamerImageModel,
    )
    from convgamer.modules.lightning_module import BackboneClassifier

    if args.ckpt_kind == "jepa":
        from convgamer.modules.jepa_module import ConvGamerVJEPAModel

        jepa = ConvGamerVJEPAModel.load_from_checkpoint(args.checkpoint, strict=False)
        encoder = jepa.encoder
        frame_encoder = getattr(encoder, "frame_encoder", None)
        if frame_encoder is None:
            raise TypeError("JEPA checkpoint encoder has no frame_encoder to export")
        native_state = frame_encoder.state_dict()
        widths = tuple(getattr(frame_encoder, "stage_widths", (96, 192, 384, 768)))
        model_cfg = {"hidden_dim": widths[0], "widths": widths}
    elif args.ckpt_kind == "image":
        image_ckpt = BackboneClassifier.load_from_checkpoint(args.checkpoint, strict=False)
        native_state = image_ckpt.model.state_dict()  # type: ignore[attr-defined]  # nn.Module
        model_cfg = dict(image_ckpt.hparams["model"])
    else:
        video_ckpt = BackboneClassifier.load_from_checkpoint(args.checkpoint, strict=False)
        model = video_ckpt.model  # type: ignore[attr-defined]
        frame_encoder = getattr(model, "frame_encoder", None)
        if frame_encoder is None:
            raise TypeError(
                f"--ckpt-kind video needs a ConvGamerEncoder checkpoint, got {type(model).__name__}"
            )
        native_state = frame_encoder.state_dict()
        model_cfg = dict(video_ckpt.hparams["model"])
    model_cfg.pop("target", None)
    model_cfg = _normalize_cfg(model_cfg)
    config_kwargs = {k: model_cfg[k] for k in _BACKBONE_KEYS if k in model_cfg}
    # ConvGamerImageConfig only consumes the image-backbone subset.
    image_keys = {
        "hidden_dim",
        "num_layers",
        "num_classes",
        "input_dim",
        "layer_scale_init",
        "mlp_ratios",
    }
    config = ConvGamerImageConfig(
        **{k: v for k, v in config_kwargs.items() if k in image_keys},
    )
    wrapper = ConvGamerImageModel(config)
    wrapper.native.load_state_dict(native_state, strict=False)
    wrapper.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print(f"Exported to {args.output_dir} (v{__version__})")


if __name__ == "__main__":
    main()
