import os

import numpy as np
import torch
from tqdm import tqdm

from args import Arguments
from UNet import build_model, infer_model_variant_from_state_dict
from datasets import get_dataloaders
from utils import AverageMeter, ScoreMeter, get_loss_fn, load_checkpoint_state


def _infer_n_classes_from_state_dict(state_dict):
    for key in ("outc.conv.weight", "module.outc.conv.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return None


def _resolve_model_and_classes(args, state_dict):
    inferred_variant = infer_model_variant_from_state_dict(state_dict)
    requested_variant = getattr(args, "model_variant", None)
    model_variant = inferred_variant if requested_variant in (None, "", "auto") else requested_variant

    inferred_n_classes = _infer_n_classes_from_state_dict(state_dict)
    requested_n_classes = getattr(args, "n_classes", None)

    if requested_n_classes is None and inferred_n_classes is not None:
        n_classes = inferred_n_classes
    elif inferred_n_classes is not None and int(requested_n_classes) != int(inferred_n_classes):
        print(
            f"Warning: n_classes={requested_n_classes} mismatches checkpoint head={inferred_n_classes}. "
            f"Using checkpoint value {inferred_n_classes}."
        )
        n_classes = inferred_n_classes
    elif requested_n_classes is not None:
        n_classes = int(requested_n_classes)
    else:
        raise ValueError(
            "Unable to determine n_classes. Provide --n_classes or use a checkpoint "
            "that contains 'outc.conv.weight'."
        )

    return model_variant, n_classes, inferred_variant


def _load_model_for_checkpoint(args):
    state_dict, _ = load_checkpoint_state(args.model_path, args.device)
    model_variant, n_classes, inferred_variant = _resolve_model_and_classes(args, state_dict)

    model = build_model(
        n_classes=n_classes,
        model_variant=model_variant,
        encoder_pretrained=False,
    ).to(args.device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Warning: strict checkpoint load failed: {e}")
        model.load_state_dict(state_dict, strict=False)

    print(
        f"Loaded checkpoint: {args.model_path}\n"
        f"- inferred_variant={inferred_variant}\n"
        f"- using_variant={model_variant}\n"
        f"- n_classes={n_classes}"
    )
    return model, n_classes, model_variant


@torch.no_grad()
def eval_epoch(model, dataloader, n_classes, criterion, device, pred_dir=None):
    model.eval()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(n_classes)

    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"

    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)

    for inputs, labels, names in tqdm(dataloader, ncols=0, leave=False, desc="eval"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        loss_meter.update(loss.item(), inputs.size(0))
        score_meter.update(preds, labels_np)

        if pred_dir:
            for i, name in enumerate(names):
                np.save(os.path.join(pred_dir, f"{os.path.splitext(name)[0]}.npy"), preds[i].astype(np.uint8))

    return loss_meter.avg, score_meter.get_scores()


def evaluate(args, mode, save_pred=False):
    _, val_loader, test_loader = get_dataloaders(args)
    if mode == "val":
        dataloader = val_loader
    elif mode == "test":
        dataloader = test_loader
    else:
        raise ValueError(f"Unsupported mode={mode}; expected 'val' or 'test'.")

    model, n_classes, _ = _load_model_for_checkpoint(args)
    criterion = get_loss_fn(args.loss_type, args.ignore_index).to(args.device)

    pred_dir = args.pred_dir if save_pred else None
    eval_loss, scores = eval_epoch(
        model=model,
        dataloader=dataloader,
        n_classes=n_classes,
        criterion=criterion,
        device=args.device,
        pred_dir=pred_dir,
    )

    miou = float(scores["mIoU"])
    acc = float(scores["accuracy"])
    ious = [float(x) for x in scores["IoUs"]]

    print(f"{mode} | loss: {eval_loss:.4f} | acc: {acc:.4f} | mIoU: {miou:.4f}")
    print(f"{mode} | IoUs per class: {ious}")

    return scores


if __name__ == "__main__":
    arg_parser = Arguments()
    arg_parser.parser.add_argument("--mode", "-m", choices=["val", "test"], required=True)
    arg_parser.parser.add_argument(
        "--save_pred",
        action="store_true",
        help="Save raw predicted masks (.npy) into args.pred_dir.",
    )
    args = arg_parser.parse_args(verbose=True)
    evaluate(args, args.mode, save_pred=args.save_pred)
