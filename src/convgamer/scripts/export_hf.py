"""Export ConvGamer model to HuggingFace format.

CLI: python -m convgamer.scripts.export_hf --checkpoint <path> --output-dir <dir>
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exported_model")
    parser.add_argument(
        "--ckpt-kind",
        type=str,
        choices=("video", "image"),
        default="video",
        help="video loads the video checkpoint's frame_encoder; image loads an image checkpoint.",
    )
    args = parser.parse_args()

    from convgamer import __version__
    from convgamer.integrations.transformers import (
        ConvGamerImageConfig,
        ConvGamerImageModel,
    )
    from convgamer.models.inception_next.encoder import InceptionNeXtEncoder
    from convgamer.modules.lightning_module import BackboneClassifier

    if args.ckpt_kind == "image":
        image_ckpt = BackboneClassifier.load_from_checkpoint(args.checkpoint, strict=False)
        native_state = image_ckpt.model.state_dict()  # type: ignore[attr-defined]  # nn.Module
        model_cfg = image_ckpt.hparams["model"]
    else:
        video_ckpt = BackboneClassifier.load_from_checkpoint(args.checkpoint, strict=False)
        frame_encoder: InceptionNeXtEncoder = video_ckpt.model.frame_encoder  # type: ignore[attr-defined]
        native_state = frame_encoder.state_dict()
        model_cfg = video_ckpt.hparams["model"]
    if isinstance(model_cfg.get("num_layers"), list):
        model_cfg["num_layers"] = tuple(model_cfg["num_layers"])
    if isinstance(model_cfg.get("mlp_ratios"), list):
        model_cfg["mlp_ratios"] = tuple(model_cfg.get("mlp_ratios", (4, 4, 4, 3)))
    config = ConvGamerImageConfig(
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_classes=model_cfg["num_classes"],
        input_dim=model_cfg["input_dim"],
        layer_scale_init=model_cfg["layer_scale_init"],
        mlp_ratios=tuple(model_cfg.get("mlp_ratios", (4, 4, 4, 3))),
    )
    wrapper = ConvGamerImageModel(config)
    wrapper.native.load_state_dict(native_state)
    wrapper.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print(f"Exported to {args.output_dir} (v{__version__})")


if __name__ == "__main__":
    main()
