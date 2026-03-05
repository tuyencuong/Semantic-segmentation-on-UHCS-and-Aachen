import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    has_module = all(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[7:]: v for k, v in state_dict.items()}


def load_checkpoint_state(checkpoint_path, map_location):
    """Load a checkpoint robustly across old/new torch serialization behaviors."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    return strip_module_prefix(state_dict), ckpt


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, weight=1):
        self.val = float(val)
        self.sum += float(val) * weight
        self.count += int(weight)
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    @property
    def average(self):
        return np.round(self.avg, 5)


class ScoreMeter:
    def __init__(self, n_classes):
        self.n_classes = int(n_classes)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)

    def update(self, pred, label):
        # pred: (N,H,W) or (N,C,H,W); label: (N,H,W)
        if pred.ndim == 4:
            pred = np.argmax(pred, axis=1)
        if label.ndim == 4 and label.shape[1] == 1:
            label = label[:, 0]

        pred = pred.astype(np.int64)
        label = label.astype(np.int64)
        n = self.n_classes

        for b in range(pred.shape[0]):
            p = pred[b].reshape(-1)
            g = label[b].reshape(-1)
            valid = (g >= 0) & (g < n) & (p >= 0) & (p < n)
            idx = g[valid] * n + p[valid]
            hist = np.bincount(idx, minlength=n * n).reshape(n, n)
            self.confusion_matrix += hist

    def get_scores(self, verbose=False):
        cm = self.confusion_matrix
        eps = 1e-8

        precision = cm.diagonal() / (cm.sum(axis=0) + eps)
        recall = cm.diagonal() / (cm.sum(axis=1) + eps)
        iou = cm.diagonal() / (cm.sum(axis=1) + cm.sum(axis=0) - cm.diagonal() + eps)
        acc = cm.diagonal().sum() / (cm.sum() + eps)
        miou = iou.mean()

        score_dict = {
            "accuracy": float(acc),
            "mIoU": float(miou),
            "IoUs": iou.astype(np.float64),
            "precision": precision.astype(np.float64),
            "recall": recall.astype(np.float64),
            "fraction_error": ((cm.sum(axis=0) - cm.sum(axis=1)) / (cm.sum() + eps)).astype(np.float64),
        }
        if verbose:
            print(f"accuracy={score_dict['accuracy']:.5f} mIoU={score_dict['mIoU']:.5f}")
        return score_dict


class Recorder:
    def __init__(self, headers):
        self.headers = list(headers)
        self.record = {header: [] for header in self.headers}

    def update(self, vals):
        for header, val in zip(self.headers, vals):
            self.record[header].append(val)

    def save(self, path):
        pd.DataFrame(self.record).to_csv(path, index=False)


class ModelSaver:
    def __init__(self, path, delta=0.0):
        self.model_path = path
        self.best_epoch = 0
        self.best_score = -np.inf
        self.delta = float(delta)

    def save_models(self, score, epoch, model, ious):
        if score > self.best_score + self.delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": state_dict,
                    "ious": ious,
                },
                self.model_path,
            )
            print(f"Validation score improved to {score:.5f}; saved model to {self.model_path}")


class LRScheduler:
    LR_SCHEDULER_MAP = {
        "CAWR": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "MultiStepLR": optim.lr_scheduler.MultiStepLR,
        "CyclicLR": optim.lr_scheduler.CyclicLR,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR,
    }
    STEP_EVERY_BATCH = ("CAWR", "CyclicLR", "OneCycleLR")
    VALID_PARAMS = {
        "CAWR": {"T_0", "T_mult", "eta_min", "last_epoch", "verbose"},
        "MultiStepLR": {"milestones", "gamma", "last_epoch", "verbose"},
        "CyclicLR": {
            "base_lr",
            "max_lr",
            "step_size_up",
            "step_size_down",
            "mode",
            "gamma",
            "scale_fn",
            "scale_mode",
            "cycle_momentum",
            "base_momentum",
            "max_momentum",
            "last_epoch",
            "verbose",
        },
        "OneCycleLR": {
            "max_lr",
            "epochs",
            "steps_per_epoch",
            "pct_start",
            "anneal_strategy",
            "cycle_momentum",
            "base_momentum",
            "max_momentum",
            "div_factor",
            "final_div_factor",
            "three_phase",
            "last_epoch",
            "verbose",
        },
    }
    REQUIRED_PARAMS = {
        "CAWR": {"T_0"},
        "MultiStepLR": {"milestones"},
        "CyclicLR": set(),
        "OneCycleLR": {"max_lr"},
    }

    def __init__(self, lr_scheduler_args, optimizer):
        self.no_scheduler = lr_scheduler_args is None
        if self.no_scheduler:
            return

        args = lr_scheduler_args
        if args.type not in self.LR_SCHEDULER_MAP:
            raise ValueError(f"unsupported lr scheduler: {args.type}")

        params = {}
        for k, v in args.params.items():
            if isinstance(v, str):
                try:
                    params[k] = ast.literal_eval(v)
                except Exception:
                    params[k] = v
            else:
                params[k] = v

        valid_keys = self.VALID_PARAMS[args.type]
        params = {k: v for k, v in params.items() if k in valid_keys}
        missing = self.REQUIRED_PARAMS[args.type] - set(params.keys())
        if missing:
            raise ValueError(f"Missing required scheduler params for {args.type}: {sorted(missing)}")

        self.lr_scheduler = self.LR_SCHEDULER_MAP[args.type](optimizer, **params)
        self.step_every_batch = args.type in self.STEP_EVERY_BATCH

    def step(self, last_batch=False):
        if self.no_scheduler:
            return
        if self.step_every_batch:
            self.lr_scheduler.step()
        elif last_batch:
            self.lr_scheduler.step()

    def state_dict(self):
        if self.no_scheduler:
            return None
        return self.lr_scheduler.state_dict()


def get_optimizer(optimizer_args, model):
    args = optimizer_args
    if isinstance(args, dict):
        encoder_lr = float(args["encoder_lr"])
        decoder_lr = float(args["decoder_lr"])
        weight_decay = float(args["weight_decay"])
        opt_type = args["type"]
    else:
        encoder_lr = float(args.encoder_lr)
        decoder_lr = float(args.decoder_lr)
        weight_decay = float(args.weight_decay)
        opt_type = args.type

    actual_model = model.module if hasattr(model, "module") else model
    if not (hasattr(actual_model, "encoder") and hasattr(actual_model, "decoder")):
        params = actual_model.parameters()
        if opt_type == "Adam":
            return optim.Adam(params, lr=decoder_lr, weight_decay=weight_decay)
        if opt_type == "AdamW":
            return optim.AdamW(params, lr=decoder_lr, weight_decay=weight_decay)
        if opt_type == "SGD":
            return optim.SGD(params, lr=decoder_lr, weight_decay=weight_decay)
        raise ValueError(f"unsupported optimizer: {opt_type}")

    if hasattr(actual_model.encoder, "__iter__"):
        encoder_params = []
        for module in actual_model.encoder:
            encoder_params.extend([p for p in module.parameters() if p.requires_grad])
    else:
        encoder_params = [p for p in actual_model.encoder.parameters() if p.requires_grad]

    if hasattr(actual_model.decoder, "__iter__"):
        decoder_params = []
        for module in actual_model.decoder:
            decoder_params.extend([p for p in module.parameters() if p.requires_grad])
    else:
        decoder_params = [p for p in actual_model.decoder.parameters() if p.requires_grad]

    list_params = [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": decoder_params, "lr": decoder_lr, "weight_decay": weight_decay},
    ]

    if opt_type == "Adam":
        return optim.Adam(list_params)
    if opt_type == "AdamW":
        return optim.AdamW(list_params)
    if opt_type == "SGD":
        return optim.SGD(list_params)
    raise ValueError(f"unsupported optimizer: {opt_type}")


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, output, target, eps=1e-8):
        b, c = output.shape[:2]
        pred = F.softmax(output, dim=1).view(b, c, -1)
        target = target.view(b, -1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask.unsqueeze(1)
            target = torch.where(mask, target, torch.zeros_like(target))
            onehot = F.one_hot(target.long(), c).permute(0, 2, 1) * mask.unsqueeze(1)
        else:
            onehot = F.one_hot(target.long(), c).permute(0, 2, 1)

        inter = (pred * onehot).sum(dim=[0, 2])
        total = (pred + onehot).sum(dim=[0, 2])
        dice = 2.0 * inter / (total + eps)

        if self.class_weights is not None:
            w = torch.as_tensor(self.class_weights, device=dice.device, dtype=dice.dtype)
            dice = dice * w
        return 1.0 - dice.mean()


class JaccardLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, eps=1e-8):
        b, c = output.shape[:2]
        pred = F.softmax(output, dim=1).view(b, c, -1)
        target = target.view(b, -1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred * mask.unsqueeze(1)
            target = torch.where(mask, target, torch.zeros_like(target))
            onehot = F.one_hot(target.long(), c).permute(0, 2, 1) * mask.unsqueeze(1)
        else:
            onehot = F.one_hot(target.long(), c).permute(0, 2, 1)

        inter = (pred * onehot).sum(dim=[0, 2])
        union = (pred + onehot).sum(dim=[0, 2]) - inter
        jaccard = inter / (union + eps)
        return 1.0 - jaccard.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_fn(loss_type, ignore_index, class_weights=None):
    if loss_type == "CE":
        weight = torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    if loss_type == "Dice":
        return DiceLoss(ignore_index=ignore_index, class_weights=class_weights)
    if loss_type == "Jaccard":
        return JaccardLoss(ignore_index=ignore_index)
    if loss_type == "Focal":
        return FocalLoss(alpha=0.25, gamma=2.0, ignore_index=ignore_index)
    raise ValueError(f"unsupported loss type: {loss_type}")

