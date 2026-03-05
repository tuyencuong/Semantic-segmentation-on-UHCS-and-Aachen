import os
import time

import torch
from tqdm import tqdm

from args import Arguments
from UNet import build_model
from datasets import get_dataloaders
from eval import eval_epoch
from utils import AverageMeter, ScoreMeter, Recorder, LRScheduler, get_optimizer, get_loss_fn


def train_epoch(model, dataloader, n_classes, optimizer, lr_scheduler, criterion, device):
    model.train()
    loss_meter = AverageMeter()
    score_meter = ScoreMeter(n_classes)

    amp_enabled = device.type == "cuda"
    amp_device = "cuda" if amp_enabled else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

    for i, (inputs, labels, _) in enumerate(tqdm(dataloader, ncols=0, leave=False, desc="train")):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(amp_device, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step(last_batch=(i == len(dataloader) - 1))

        preds = outputs.detach().argmax(dim=1).cpu().numpy()
        score_meter.update(preds, labels.cpu().numpy())
        loss_meter.update(loss.item(), inputs.size(0))

    scores = score_meter.get_scores()
    return loss_meter.avg, scores


def _save_checkpoint(path, epoch, model, optimizer, lr_scheduler, args, best_metric):
    state = {
        "epoch": int(epoch),
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict() if hasattr(lr_scheduler, "state_dict") else None,
        "best_metric": float(best_metric),
        "model_variant": getattr(args, "model_variant", None),
        "n_classes": int(args.n_classes),
        "dataset": getattr(args, "dataset", None),
    }
    torch.save(state, path)


def train(args):
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    Arguments.save_args(args, args.args_path)

    train_loader, val_loader, _ = get_dataloaders(args)

    model = build_model(
        n_classes=args.n_classes,
        model_variant=args.model_variant,
        encoder_pretrained=args.encoder_pretrained,
    ).to(args.device)

    if getattr(args, "multi_gpu", False) and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.optimizer.encoder_lr
    decoder_lr = args.decoder_lr if args.decoder_lr is not None else args.optimizer.decoder_lr

    optimizer = get_optimizer(
        {
            "type": args.optimizer.type,
            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
            "weight_decay": args.optimizer.weight_decay,
        },
        model,
    )
    lr_scheduler = LRScheduler(getattr(args, "lr_scheduler", None), optimizer)
    criterion = get_loss_fn(args.loss_type, args.ignore_index)

    headers = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_miou",
        "val_loss",
        "val_acc",
        "val_miou",
    ]
    for i in range(args.n_classes):
        headers.append(f"train_iou_class_{i}")
    for i in range(args.n_classes):
        headers.append(f"val_iou_class_{i}")
    recorder = Recorder(headers)

    model_path = args.model_path
    best_model_path = os.path.join(args.checkpoints_dir, "best_model.pth")

    save_every = int(getattr(args, "save_every", 0) or 0)
    best_val_miou = -1.0
    start_time = time.time()

    txt_log_path = os.path.join(args.checkpoints_dir, "train_results.txt")
    with open(txt_log_path, "w", encoding="utf-8") as f:
        f.write("Epoch | TrainLoss | TrainAcc | TrainmIoU | ValLoss | ValAcc | ValmIoU\n")
        f.write("-" * 80 + "\n")

    for epoch in range(1, args.n_epochs + 1):
        print(f"\n{args.experim_name} Epoch {epoch}/{args.n_epochs}")

        train_loss, train_scores = train_epoch(
            model=model,
            dataloader=train_loader,
            n_classes=args.n_classes,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=args.device,
        )

        val_loss, val_scores = eval_epoch(
            model=model,
            dataloader=val_loader,
            n_classes=args.n_classes,
            criterion=criterion,
            device=args.device,
            pred_dir=None,
        )

        train_miou = float(train_scores["mIoU"])
        train_acc = float(train_scores["accuracy"])
        val_miou = float(val_scores["mIoU"])
        val_acc = float(val_scores["accuracy"])
        train_ious = [float(x) for x in train_scores["IoUs"]]
        val_ious = [float(x) for x in val_scores["IoUs"]]

        print(
            f"train | loss: {train_loss:.4f} | acc: {train_acc:.4f} | mIoU: {train_miou:.4f} | IoUs: {train_ious}"
        )
        print(f"valid | loss: {val_loss:.4f} | acc: {val_acc:.4f} | mIoU: {val_miou:.4f} | IoUs: {val_ious}")

        row = [epoch, train_loss, train_acc, train_miou, val_loss, val_acc, val_miou] + train_ious + val_ious
        recorder.update(row)

        with open(txt_log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch:5d} | {train_loss:9.4f} | {train_acc:8.4f} | {train_miou:9.4f} | "
                f"{val_loss:7.4f} | {val_acc:6.4f} | {val_miou:7.4f}\n"
            )

        # Keep model.pth as latest checkpoint to preserve older workflow expectation.
        _save_checkpoint(model_path, epoch, model, optimizer, lr_scheduler, args, best_val_miou)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            _save_checkpoint(best_model_path, epoch, model, optimizer, lr_scheduler, args, best_val_miou)
            print(f"Best model updated at epoch {epoch} (mIoU={best_val_miou:.4f})")

        if save_every > 0 and (epoch % save_every == 0):
            periodic_path = os.path.join(args.checkpoints_dir, f"checkpoint_epoch_{epoch}.pth")
            _save_checkpoint(periodic_path, epoch, model, optimizer, lr_scheduler, args, best_val_miou)

        recorder.save(args.record_path)

    elapsed_min = (time.time() - start_time) / 60.0
    print(f"\nTraining complete in {elapsed_min:.2f} min")
    print(f"Best validation mIoU: {best_val_miou:.4f}")
    print(f"Latest checkpoint: {model_path}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Metrics CSV: {args.record_path}")


if __name__ == "__main__":
    arg_parser = Arguments()
    arg_parser.parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Save periodic checkpoints every N epochs (0/None disables periodic saves).",
    )
    arg_parser.parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable DataParallel across all visible GPUs (disabled by default).",
    )
    args = arg_parser.parse_args(verbose=True)
    train(args)
