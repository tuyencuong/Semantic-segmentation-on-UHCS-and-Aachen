import argparse
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from UNet import build_model, infer_model_variant_from_state_dict
from utils import load_checkpoint_state


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _infer_n_classes_from_state_dict(state_dict):
    for key in ("outc.conv.weight", "module.outc.conv.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return None


def _resolve_model_and_classes(model_variant, n_classes, state_dict):
    inferred_variant = infer_model_variant_from_state_dict(state_dict)
    use_variant = inferred_variant if model_variant in (None, "", "auto") else model_variant

    inferred_n_classes = _infer_n_classes_from_state_dict(state_dict)
    if n_classes is None and inferred_n_classes is not None:
        use_n_classes = inferred_n_classes
    elif inferred_n_classes is not None and int(n_classes) != int(inferred_n_classes):
        print(
            f"Warning: n_classes={n_classes} mismatches checkpoint head={inferred_n_classes}. "
            f"Using checkpoint value {inferred_n_classes}."
        )
        use_n_classes = inferred_n_classes
    elif n_classes is not None:
        use_n_classes = int(n_classes)
    else:
        raise ValueError(
            "Unable to determine n_classes. Provide --n_classes or use a checkpoint "
            "that contains 'outc.conv.weight'."
        )

    return use_variant, use_n_classes, inferred_variant


class MicrostructureDataset(Dataset):
    def __init__(self, image_dir, image_size=None):
        self.image_dir = image_dir
        self.image_paths = sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
        )
        self.image_size = tuple(image_size) if image_size is not None else None
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_rgb = rgb.copy()
        if self.image_size is not None:
            w, h = self.image_size
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).float()

        return x, os.path.basename(path), orig_rgb


def _segment_collate(batch):
    inputs, names, orig_rgbs = zip(*batch)
    input_shapes = [tuple(x.shape) for x in inputs]
    if len(set(input_shapes)) != 1:
        raise ValueError(
            "Input tensors have different shapes. Use --batch_size 1 or set --image_size W H "
            "to force a common model input size."
        )
    return torch.stack(inputs, dim=0), list(names), list(orig_rgbs)


def default_color_map(n_classes):
    if n_classes == 2:
        return {
            0: (0, 0, 0),
            1: (0, 255, 0),
        }
    if n_classes == 4:
        # UHCS class colors (match legacy visuals):
        # 0=ferritic matrix, 1=class1, 2=class2, 3=class3
        return {
            0: (54, 1, 84),      # Purple (ferritic matrix)
            1: (49, 102, 141),   # Dark blue
            2: (248, 230, 33),   # Yellow
            3: (255, 99, 71),    # Orange-red
        }

    base = [
        (0, 0, 0),
        (49, 102, 141),
        (248, 230, 33),
        (255, 99, 71),
        (44, 160, 44),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
    ]
    return {i: base[i % len(base)] for i in range(n_classes)}


def colorize(mask, color_map):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in color_map.items():
        out[mask == cls_id] = rgb
    return out


def overlay_mask_only(image_rgb, mask_rgb, seg, alpha=0.55, background_class=0):
    """
    Blend colors only on predicted mask pixels and keep non-mask pixels unchanged.
    This avoids darkening the full image.
    """
    overlay = image_rgb.copy()
    fg = seg != background_class
    if np.any(fg):
        blended = (
            image_rgb.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha
        ).astype(np.uint8)
        overlay[fg] = blended[fg]
    return overlay


def remove_small_regions(seg, min_area=0):
    if min_area <= 0:
        return seg
    filtered = seg.copy()
    for class_idx in np.unique(seg):
        if class_idx == 0:
            continue
        class_mask = (seg == class_idx).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                filtered[labels == label_id] = 0
    return filtered


@torch.no_grad()
def segment_images(model, dataloader, device, output_dir, n_classes, min_area=0, alpha=0.55, save_npy=True, verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    color_map = default_color_map(n_classes)

    model.eval()
    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"

    for inputs, names, orig_rgbs in tqdm(dataloader, desc="segment", ncols=0):
        inputs = inputs.to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            logits = model(inputs)

        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for i, name in enumerate(names):
            base = os.path.splitext(name)[0]
            img_rgb = np.ascontiguousarray(orig_rgbs[i])
            seg = preds[i]
            # Always restore prediction to original input image size.
            if seg.shape[:2] != img_rgb.shape[:2]:
                seg = cv2.resize(
                    seg.astype(np.uint8),
                    (img_rgb.shape[1], img_rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            seg = remove_small_regions(seg, min_area=min_area)

            mask_rgb = colorize(seg, color_map)
            overlay = overlay_mask_only(img_rgb, mask_rgb, seg, alpha=alpha, background_class=0)

            cv2.imwrite(os.path.join(output_dir, f"{base}_input.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(output_dir, f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if save_npy:
                np.save(os.path.join(output_dir, f"{base}_mask.npy"), seg)

            if verbose:
                cls_counts = {int(c): int((seg == c).sum()) for c in np.unique(seg)}
                print(f"{base}: {cls_counts}")


def load_model_from_checkpoint(model_path, device, model_variant=None, n_classes=None):
    state_dict, _ = load_checkpoint_state(model_path, device)
    use_variant, use_n_classes, inferred_variant = _resolve_model_and_classes(model_variant, n_classes, state_dict)

    model = build_model(
        n_classes=use_n_classes,
        model_variant=use_variant,
        encoder_pretrained=False,
    ).to(device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: strict checkpoint load failed: {e}")
        model.load_state_dict(state_dict, strict=False)

    print(
        f"Loaded model from {model_path}\n"
        f"- inferred_variant={inferred_variant}\n"
        f"- using_variant={use_variant}\n"
        f"- n_classes={use_n_classes}"
    )

    return model, use_n_classes


def main():
    parser = argparse.ArgumentParser(description="Unified segmentation inference for Aachen and UHCS checkpoints.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint path (.pth/.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for segmentation outputs")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_variant", type=str, default="auto", help="Model variant or 'auto'")
    parser.add_argument("--n_classes", type=int, default=None, help="Override number of output classes")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="Resize to (W H); default keeps original")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--min_area", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_save_npy", action="store_true", help="Disable saving *_mask.npy")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"image_dir not found: {args.image_dir}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, n_classes = load_model_from_checkpoint(
        model_path=args.model_path,
        device=device,
        model_variant=args.model_variant,
        n_classes=args.n_classes,
    )

    dataset = MicrostructureDataset(args.image_dir, image_size=args.image_size)
    if len(dataset) == 0:
        raise RuntimeError(f"No supported images found in {args.image_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_segment_collate,
    )

    segment_images(
        model=model,
        dataloader=dataloader,
        device=device,
        output_dir=args.output_dir,
        n_classes=n_classes,
        min_area=args.min_area,
        alpha=args.alpha,
        save_npy=not args.no_save_npy,
        verbose=args.verbose,
    )

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
