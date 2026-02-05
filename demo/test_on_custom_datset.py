#!/usr/bin/env python3
"""GBPG-Net Inference Script for Custom Datasets.

This script provides inference capabilities for the GBPG-Net model on custom
image datasets. It supports both single image and batch processing, with
optional PSNR/SSIM metric computation when ground truth images are available.

Usage as module:
    from demo.test_on_custom_datset import main

    # Basic inference
    main(
        weights_path="./weights/rain13k.pth",
        input_path="./input",
        output_dir="./output"
    )

    # Evaluation with metrics
    result = main(
        weights_path="./weights/rain13k.pth",
        input_path="./datasets/test/input",
        gt_path="./datasets/test/gt",
        output_dir="./output"
    )
    print(f"PSNR: {result['psnr']}, SSIM: {result['ssim']}")

    # Using predefined dataset
    main(
        weights_path="./weights/rain13k.pth",
        task="Derain",
        dataset="Rain13K",
        output_dir="./output"
    )

Usage from command line:
    python test_on_custom_datset.py \\
        --weights ./weights/rain13k.pth \\
        --input ./datasets/test/input \\
        --gt ./datasets/test/gt \\
        --output-dir ./results

Author: GBPG-Net Contributors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
# ROOT_DIR = Path(__file__).parent.parent
# sys.path.insert(0, str(ROOT_DIR))

from basicsr.metrics.psnr_ssim_compute import calculate_psnr_ssim_auto
from demo.utils import (
    AverageMeter,
    _get_paths_from_images,
    load,
    tensor2uint8,
    uint2tensor,
)


def extract_hue_channel(
    img_bgr: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Extract hue channel from BGR image and convert to tensor.

    The hue channel in HSV color space is robust to rain and snow
    disturbances, making it an effective prior for image restoration.

    Args:
        img_bgr: Input image in BGR format with values in [0, 255].
        device: PyTorch device to place the tensor on.

    Returns:
        Hue channel as a tensor of shape (1, 1, H, W) with values in [0, 1].
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_h = img_hsv[:, :, 0:1]  # Extract hue channel, keep H, W dimensions
    img_h = img_h.astype(np.float32) / 255.0
    img_h_tensor = torch.from_numpy(img_h).permute(2, 0, 1).unsqueeze(0).to(device)
    return img_h_tensor


def load_model(
    weights_path: str | Path,
    device: torch.device,
    key: str = "params",
    delete_module: bool = False,
    delete_str: str = "module.",
) -> torch.nn.Module:
    """Load GBPG-Net model with pretrained weights.

    Args:
        weights_path: Path to the pretrained weights file (.pth).
        device: PyTorch device to load the model on.
        key: Key to look for in the checkpoint file. Defaults to "params".
        delete_module: Whether to delete the module prefix from keys. Defaults to False.
        delete_str: String prefix to delete from keys when delete_module is True.
            Defaults to "module.".

    Returns:
        Loaded model in evaluation mode.
    """
    from basicsr.archs.GBPGNet_arch import GBPGNet
    # from basicsr.archs.GBPGNet_arch_bf import GBPGNet

    model = GBPGNet(depths=[1, 3, 3, 4], embed_dims=32)
    load(str(weights_path), model, key=key, delete_module=delete_module, delete_str=delete_str)

    model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def process_single_image(
    model: torch.nn.Module,
    lr_image_path: str | Path,
    device: torch.device,
) -> np.ndarray:
    """Process a single input image through the model.

    Args:
        model: GBPG-Net model.
        lr_image_path: Path to the low-quality (rainy/snowy) input image.
        device: PyTorch device for computation.

    Returns:
        Restored image as a NumPy array in RGB format with values in [0, 255].
    """
    # Read image (BGR format, HWC)
    img_bgr = cv2.imread(str(lr_image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {lr_image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to tensors
    rgb_tensor = uint2tensor(img_rgb).to(device)
    hue_tensor = extract_hue_channel(img_bgr, device)

    # Inference
    output = model(rgb_tensor, hue_tensor)
    output = output[0] if isinstance(output, tuple) else output

    # Convert back to numpy
    output_rgb = tensor2uint8(output)

    return output_rgb


@torch.inference_mode()
def evaluate_dataset(
    model: torch.nn.Module,
    lr_path: str | Path,
    gt_path: str | Path | None,
    device: torch.device,
    crop_border: int = 0,
    test_y_channel: bool = False,
    save: bool = False,
    output_dir: str | Path | None = None,
    input_suffix: str = "",
    gt_suffix: str = "",
    save_suffix: str = "",
    verbose: bool = True,
) -> dict[str, float] | None:
    """Evaluate model on a dataset with optional ground truth comparison.

    Args:
        model: GBPG-Net model.
        lr_path: Path to low-quality (input) images or directory.
        gt_path: Path to ground truth images or directory. If None, only
            inference is performed without metrics.
        device: PyTorch device for computation.
        crop_border: Number of pixels to crop from borders before evaluation.
        test_y_channel: Whether to test on Y channel only (for perceptual quality).
        save: Whether to save output images.
        output_dir: Directory to save output images.
        input_suffix: Suffix to filter input images.
        gt_suffix: Suffix to filter ground truth images.
        save_suffix: Suffix to save output images.
        verbose: Whether to print progress.

    Returns:
        Dictionary with 'psnr' and 'ssim' keys if gt_path is provided,
        otherwise None.
    """
    # Handle input paths
    if os.path.isdir(str(lr_path)):
        lr_images = _get_paths_from_images(str(lr_path), suffix=input_suffix)
    else:
        lr_images = [str(lr_path)]

    gt_images = None
    if gt_path is not None:
        if os.path.isdir(str(gt_path)):
            gt_images = _get_paths_from_images(str(gt_path), suffix=gt_suffix)
        else:
            gt_images = [str(gt_path)]

        if len(lr_images) != len(gt_images):
            raise ValueError(
                f"Number of LR images ({len(lr_images)}) does not match "
                f"number of GT images ({len(gt_images)})"
            )

    # Setup output directory
    if save and output_dir is not None:
        os.makedirs(str(output_dir), exist_ok=True)

    # Metrics tracking
    psnr_meter = AverageMeter() if gt_images is not None else None
    ssim_meter = AverageMeter() if gt_images is not None else None

    # Process images
    iterator = zip(lr_images, gt_images if gt_images else [None] * len(lr_images))
    for idx, (lr_path_item, gt_path_item) in enumerate(tqdm(iterator, desc="Processing", disable=not verbose)):
        base_name, ext = os.path.splitext(os.path.basename(lr_path_item))

        # Read LR image
        lr_img_bgr = cv2.imread(str(lr_path_item))
        if lr_img_bgr is None:
            print(f"Warning: Failed to read {lr_path_item}, skipping...")
            continue

        lr_img_rgb = cv2.cvtColor(lr_img_bgr, cv2.COLOR_BGR2RGB)

        # Prepare tensors
        rgb_tensor = uint2tensor(lr_img_rgb).to(device)
        hue_tensor = extract_hue_channel(lr_img_bgr, device)

        # Inference
        output = model(rgb_tensor, hue_tensor)
        output = output[0] if isinstance(output, tuple) else output

        # Save output
        if save and output_dir is not None:
            output_rgb = tensor2uint8(output)
            output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(str(output_dir), f"{base_name}{save_suffix}{ext}"), output_bgr)

        # Compute metrics if ground truth is available
        if gt_images is not None and gt_path_item is not None:
            gt_img_rgb = cv2.imread(str(gt_path_item))[:, :, ::-1]
            if gt_img_rgb is None:
                print(f"Warning: Failed to read GT {gt_path_item}, skipping metrics...")
                continue

            # gt_img_rgb = cv2.cvtColor(gt_img_bgr, cv2.COLOR_BGR2RGB)
            # gt_tensor = uint2tensor(gt_img_rgb).to(device)

            psnr_val, ssim_val = calculate_psnr_ssim_auto(
                output,
                gt_img_rgb,
                crop_border=crop_border,
                test_y=test_y_channel,
                input_order="HWC",
                color_order="RGB",
                mode="np",
            )

            if psnr_meter is not None:
                psnr_meter.update(psnr_val)
            if ssim_meter is not None:
                ssim_meter.update(ssim_val)

            if verbose:
                print(
                    f"[{idx + 1}/{len(lr_images)}] {base_name}: "
                    f"PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}"
                )

    # Print and return results
    if gt_images is not None and psnr_meter is not None and ssim_meter is not None:
        avg_psnr = psnr_meter.avg
        avg_ssim = ssim_meter.avg
        print(f"\n{'=' * 50}")
        print(f"Dataset: {len(lr_images)} images")
        print(f"Average PSNR: {avg_psnr:.4f} dB | Average SSIM: {avg_ssim:.4f}")
        print(f"{'=' * 50}")
        return {"psnr": avg_psnr, "ssim": avg_ssim}

    return None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="GBPG-Net Inference Script for Rain/Snow Removal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Path to input image(s) or directory",
    )
    parser.add_argument(
        "-g",
        "--gt",
        type=str,
        default=None,
        help="Path to ground truth image(s) or directory (optional, for metrics)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save output images",
    )

    # Model arguments
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        required=True,
        help="Path to model weights file (.pth)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="params",
        help="Key to look for in the checkpoint file",
    )
    parser.add_argument(
        "--delete-module",
        action="store_true",
        help="Whether to delete the module prefix from keys",
    )
    parser.add_argument(
        "--delete-str",
        type=str,
        default="module.",
        help="String prefix to delete from keys when --delete-module is used",
    )

    # Dataset arguments
    parser.add_argument(
        "--task",
        type=str,
        choices=["Derain", "Desnow"],
        default=None,
        help="Predefined task name (uses built-in dataset config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Predefined dataset name (uses built-in dataset config)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Number of pixels to crop from borders for evaluation",
    )
    parser.add_argument(
        "--test-y-channel",
        action="store_true",
        help="Test on Y channel only for perceptual metrics",
    )

    # Other arguments
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save output images",
    )
    parser.add_argument(
        "--input-suffix",
        type=str,
        default="",
        help="Suffix to filter input images",
    )
    parser.add_argument(
        "--gt-suffix",
        type=str,
        default="",
        help="Suffix to filter ground truth images",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    return parser.parse_args()


def main(
    weights_path: str | Path,
    input_path: str | Path | None = None,
    target_path: str | Path | None = None,
    output_dir: str | Path = "./results",
    key: str = "params",
    delete_module: bool = False,
    delete_str: str = "module.",
    task: str | None = None,
    dataset: str | None = None,
    crop_border: int = 0,
    test_y_channel: bool = False,
    save: bool = True,
    input_suffix: str = "",
    gt_suffix: str = "",
    save_suffix: str = "",
    verbose: bool = True,
) -> dict[str, float] | None:
    """Main entry point for the inference script.

    Args:
        weights_path: Path to model weights file (.pth).
        input_path: Path to input image(s) or directory.
        gt_path: Path to ground truth image(s) or directory (optional, for metrics).
        output_dir: Directory to save output images.
        key: Key to look for in the checkpoint file.
        delete_module: Whether to delete the module prefix from keys.
        delete_str: String prefix to delete from keys when delete_module is True.
        task: Predefined task name (Derain/Desnow).
        dataset: Predefined dataset name.
        crop_border: Number of pixels to crop from borders for evaluation.
        test_y_channel: Whether to test on Y channel only.
        save: Whether to save output images.
        input_suffix: Suffix to filter input images.
        gt_suffix: Suffix to filter ground truth images.
        save_suffix: Suffix to filter output images.
        verbose: Whether to print progress.

    Returns:
        Dictionary with 'psnr' and 'ssim' keys if gt_path is provided,
        otherwise None.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")

    # Load model
    if verbose:
        print(f"Loading model from: {weights_path}")
    model = load_model(
        weights_path=weights_path,
        device=device,
        key=key,
        delete_module=delete_module,
        delete_str=delete_str,
    )
    if verbose:
        print("Model loaded successfully")

    # Determine input/output paths
    if task and dataset:
        # Use predefined dataset configuration
        from basicsr.data.datapath import test_dataset_dict

        dataset_paths = test_dataset_dict(task=task, dataset=dataset)
        root_dir = Path(__file__).parent.parent
        datasets_root = root_dir / "datasets"
        lr_path = datasets_root / dataset_paths[0]
        gt_path = datasets_root / dataset_paths[1]
        if verbose:
            print(f"Using predefined dataset: {task}/{dataset}")
    else:
        # Use provided paths
        lr_path = Path(input_path) if input_path else None
        gt_path = Path(target_path) if target_path else None

    if lr_path is None:
        raise ValueError(
            "Either input_path or both task and dataset must be specified"
        )

    # Run evaluation
    return evaluate_dataset(
        model=model,
        lr_path=lr_path,
        gt_path=gt_path,
        device=device,
        crop_border=crop_border,
        test_y_channel=test_y_channel,
        save=save,
        output_dir=output_dir,
        input_suffix=input_suffix,
        gt_suffix=gt_suffix,
        save_suffix=save_suffix,
        verbose=verbose,
    )


def cli() -> None:
    """Command-line interface wrapper for main()."""
    args = parse_args()

    result = main(
        weights_path=args.weights,
        input_path=args.input,
        target_path=args.gt,
        output_dir=args.output_dir,
        key=args.key,
        delete_module=args.delete_module,
        delete_str=args.delete_str,
        task=args.task,
        dataset=args.dataset,
        crop_border=args.crop_border,
        test_y_channel=args.test_y_channel,
        save=not args.no_save,
        input_suffix=args.input_suffix,
        gt_suffix=args.gt_suffix,
        verbose=not args.quiet,
    )

    if result is not None:
        print(f"\nPSNR: {result['psnr']:.4f} dB | SSIM: {result['ssim']:.4f}")
    print("\nDone!")


if __name__ == "__main__":
    # ==================== Mode Selection ====================
    # Set True to use CLI with argparse, False to use manual config
    USE_CLI_MODE = False
    # ========================================================

    if USE_CLI_MODE:
        # Mode 1: Command-line interface with argparse
        # Run with: python test_on_custom_datset.py --weights ./weights/model.pth --input ./input
        cli()
    else:
        # Mode 2: Manual configuration (directly call main with parameters)
        result = main(
            # Required: Path to model weights
            weights_path="../pretrained/GBPGNet_Real-Rain-1K-H.pth.pth",

            # Option 1: Specify input/output paths directly
            input_path=r"E:\Dataset\Restoration\Derain\RealRain1k\RealRain-1k-H\test\input",
            target_path=r"E:\Dataset\Restoration\Derain\RealRain1k\RealRain-1k-H\test\target",
            output_dir="../outputs/results",

            # Option 2: Or use predefined dataset (uncomment and comment Option 1)
            # task="Derain",
            # dataset="Cityscapes100",
            # output_dir="./outputs/results",

            # Evaluation settings
            crop_border=0,
            test_y_channel=True,  # Rain - True, Snow - False
            save=False,
            save_suffix="",
            input_suffix="",
            gt_suffix="",
            verbose=False,
            key="params",  # params | state_dict
            delete_module=False,
            delete_str="module.",
        )

        # if result is not None:
        #     print(f"\nFinal Results: PSNR={result['psnr']:.4f} dB | SSIM={result['ssim']:.4f}")


"""
Rain100L: Average PSNR: 38.6118 dB | Average SSIM: 0.9764
Rain100H: Average PSNR: 31.0523 dB | Average SSIM: 0.9013
Test100: Average PSNR: 31.8189 dB | Average SSIM: 0.9208
Test1200: Average PSNR: 32.7061 dB | Average SSIM: 0.9251
Test2800: Average PSNR: 33.9288 dB | Average SSIM: 0.9418

RealRain1K-L: Average PSNR: 42.2149 dB | Average SSIM: 0.9890   Average PSNR: 41.9110 dB | Average SSIM: 0.9889
RealRain1K-H: Average PSNR: 40.7944 dB | Average SSIM: 0.9820

CSD: Average PSNR: 37.9497 dB | Average SSIM: 0.9810
SnowKITTI-L: Average PSNR: 37.0452 dB | Average SSIM: 0.9779
SnowCityscapes-L: Average PSNR: 39.5644 dB | Average SSIM: 0.9836
Snow100K-L: Average PSNR: 31.4183 dB | Average SSIM: 0.9110



"""