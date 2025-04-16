import os
import argparse
import torch
from omegaconf import OmegaConf
from pathlib import Path

from models import get_model
from data.transforms import get_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=Path, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="path to model checkpoint file",
    )
    parser.add_argument("--batch", "-b", type=int, required=True, help="amount of samples")
    parser.add_argument("--out", "-o", type=Path, required=False, help="output directory", default="samples")
    args = parser.parse_args()
    config_path = str(args.path)
    checkpoint_path = str(args.checkpoint) if args.checkpoint is not None else None
    assert checkpoint_path != None # silence LSP error

    batch_size = args.batch
    out_path = args.out

    config = OmegaConf.load(config_path)

    model_type = get_model(config.model.get("model_type"))
    model = model_type.load_from_checkpoint(checkpoint_path, **config.model.get("params", dict()))

    transforms = OmegaConf.to_object(config.model["output_transforms"])
    assert isinstance(transforms, list)
    model.output_transform = get_transform(transforms)

    if torch.cuda.is_available():
        model.to(torch.device("cuda"))

    model.eval()
    midi_contents = model.sample_midi(batch_size)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i, sample in enumerate(midi_contents):
        try:
            with open(os.path.join(out_path, f"sample_{i}.mid"), "wb") as f:
                f.write(sample)
        except Exception:
            print(f"Invalid format of sample {i}")
