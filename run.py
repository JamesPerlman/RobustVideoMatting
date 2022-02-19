import argparse
from inference import convert_video
from model import MattingNetwork
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description='RobustVideoMatting')
parser.add_argument('-i', '--input', type=str, required=True)
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--foreground', action='store_true')
parser.add_argument('--comp', action='store_true')
parser.add_argument('--png', action='store_true')
parser.add_argument('--mbps', type=int, default=4)

args = parser.parse_args()

input_path = Path(args.input)
output_path = input_path

output_alpha_path = input_path.parent / f"{input_path.stem}-alpha{'' if args.png else input_path.suffix}"
output_comp_path = input_path.parent / f"{input_path.stem}-comp{'' if args.png else input_path.suffix}"
output_fg_path = input_path.parent / f"{input_path.stem}-fg{'' if args.png else input_path.suffix}"

model = MattingNetwork('resnet50')
checkpoint = Path("./models/rvm_resnet50.pth")
state_dict = torch.load(checkpoint)
model.load_state_dict(state_dict)

convert_video(
    # The model, can be on any device (cpu or cuda).
    model,
    # A video file or an image sequence directory.
    input_source=input_path,
    # Choose "video" or "png_sequence"
    output_type=('png_sequence' if args.png else 'video'),
    # File path if video; directory path if png sequence.
    output_composition=(str(output_comp_path) if args.comp else None),
    # [Optional] Output the raw alpha prediction.
    output_alpha=(str(output_alpha_path) if args.alpha else None),
    # [Optional] Output the raw foreground prediction.
    output_foreground=(str(output_fg_path) if args.foreground else None),
    # Output video mbps. Not needed for png sequence.
    output_video_mbps=args.mbps,
    # A hyperparameter to adjust or use None for auto.
    downsample_ratio=None,
    # Process n frames at once for better parallelism.
    seq_chunk=12,
)
