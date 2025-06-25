import sys
from PIL import Image
import numpy as np
import os

# Usage: python side_by_side_compare.py <rendered.png> <output.png> --gt_idx <index> [--gt_dir <dir>]
#    or: python side_by_side_compare.py <ground_truth.png> <rendered.png> <output.png>

def get_gt_path(gt_arg, gt_idx=None, gt_dir=None):
    if gt_idx is not None:
        # Default directory if not provided
        if gt_dir is None:
            gt_dir = 'data/lego/train'
        gt_path = os.path.join(gt_dir, f"r_{gt_idx}.png")
        if not os.path.exists(gt_path):
            print(f"Error: Ground truth image {gt_path} does not exist.")
            sys.exit(1)
        return gt_path
    else:
        if not os.path.exists(gt_arg):
            print(f"Error: Ground truth image {gt_arg} does not exist.")
            sys.exit(1)
        return gt_arg

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a side-by-side comparison of ground truth and rendered images.")
    parser.add_argument('rendered', type=str, help='Path to rendered image')
    parser.add_argument('output', type=str, help='Path to save side-by-side image')
    parser.add_argument('--gt_idx', type=int, default=None, help='Index of ground truth image (e.g., 10 for r_10.png)')
    parser.add_argument('--gt_dir', type=str, default=None, help='Directory for ground truth images (default: data/lego/train)')
    parser.add_argument('--gt_path', type=str, default=None, help='Explicit path to ground truth image (overrides gt_idx)')
    args = parser.parse_args()

    if args.gt_path:
        gt_path = get_gt_path(args.gt_path)
    elif args.gt_idx is not None:
        gt_path = get_gt_path(None, args.gt_idx, args.gt_dir)
    else:
        print("Error: You must provide either --gt_path or --gt_idx.")
        sys.exit(1)

    if not isinstance(gt_path, str) or not gt_path:
        print("Error: Computed ground truth path is invalid.")
        sys.exit(1)

    pred = Image.open(args.rendered)
    gt = Image.open(gt_path)
    pred = pred.resize(gt.size)
    side_by_side = np.concatenate([np.array(gt), np.array(pred)], axis=1)
    Image.fromarray(side_by_side).save(args.output)
    print(f"Saved side-by-side comparison to {args.output}") 