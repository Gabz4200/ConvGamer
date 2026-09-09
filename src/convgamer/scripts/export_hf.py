from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exported_model")
    args = parser.parse_args()

    from convgamer import __version__
    from convgamer.integrations.transformers import (
        ConvGamerConfig,
        ConvGamerModel,
    )
    from convgamer.modules.lightning_module import ConvGamerModel as PLModel

    ckpt = PLModel.load_from_checkpoint(args.checkpoint)
    config = ConvGamerConfig(
        hidden_dim=ckpt.hparams["model"]["hidden_dim"],
        num_layers=ckpt.hparams["model"]["num_layers"],
        num_classes=ckpt.hparams["model"]["num_classes"],
        input_dim=ckpt.hparams["model"]["input_dim"],
        layer_scale_init=ckpt.hparams["model"]["layer_scale_init"],
    )
    wrapper = ConvGamerModel(config)
    wrapper.native.load_state_dict(ckpt.model.state_dict())
    wrapper.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    print(f"Exported to {args.output_dir} (v{__version__})")


if __name__ == "__main__":
    main()
