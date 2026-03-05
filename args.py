from argparse import ArgumentParser, Namespace
import os
from os.path import splitext

import numpy as np
import torch
import yaml


def _load_yaml_if_exists(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def _as_namespace(obj):
    if isinstance(obj, Namespace):
        return obj
    if isinstance(obj, dict):
        return Namespace(**obj)
    raise TypeError(f"Expected dict/Namespace, got {type(obj)}")


def _coerce_seq(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


class Arguments:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--dataset", default="aachen", choices=["aachen", "uhcs"])
        parser.add_argument("--config", default="default.yaml")
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)

        # Common overrides
        parser.add_argument("--output_dir", type=str, default=None)
        parser.add_argument("--record_path", type=str, default=None)
        parser.add_argument("--experim_name", type=str, default=None)
        parser.add_argument("--model_variant", type=str, default=None)
        parser.add_argument("--model_path", type=str, default=None)

        parser.add_argument("--dataset_root", type=str, default=None)
        parser.add_argument("--img_dir", type=str, default=None)
        parser.add_argument("--label_dir", type=str, default=None)

        parser.add_argument("--n_classes", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=None)
        parser.add_argument("--n_epochs", type=int, default=None)
        parser.add_argument("--loss_type", type=str, default=None)
        parser.add_argument("--ignore_index", type=int, default=None)

        parser.add_argument("--encoder_lr", type=float, default=None)
        parser.add_argument("--decoder_lr", type=float, default=None)

        parser.add_argument(
            "--encoder_pretrained",
            action="store_true",
            default=None,
            help="Use ImageNet-pretrained VGG16 encoder weights when building a fresh model.",
        )
        parser.add_argument(
            "--no_encoder_pretrained",
            action="store_true",
            default=None,
            help="Disable pretrained VGG16 encoder weights.",
        )
        self.parser = parser

    def parse_args(self, verbose=False, use_random_seed=True):
        cli = self.parser.parse_args()

        repo_root = os.path.dirname(os.path.abspath(__file__))
        default_config_path = os.path.join(repo_root, "configs", cli.dataset, "default.yaml")
        config_path = os.path.join(repo_root, "configs", cli.dataset, cli.config)

        merged = {}
        merged.update(_load_yaml_if_exists(default_config_path))
        if cli.config != "default.yaml":
            merged.update(_load_yaml_if_exists(config_path))

        # Apply CLI only when meaningful to avoid overwriting config values by None.
        cli_dict = vars(cli)
        always_keys = {"dataset", "config", "gpu_id", "seed"}
        for k, v in cli_dict.items():
            if k in always_keys:
                merged[k] = v
            elif v is not None:
                merged[k] = v

        args = Namespace(**merged)

        # Ensure optional fields exist even when absent from both config and CLI override.
        optional_defaults = {
            "output_dir": None,
            "record_path": None,
            "experim_name": None,
            "model_variant": None,
            "model_path": None,
            "dataset_root": None,
            "img_dir": None,
            "label_dir": None,
            "n_classes": None,
            "batch_size": None,
            "n_epochs": None,
            "loss_type": None,
            "ignore_index": None,
            "encoder_lr": None,
            "decoder_lr": None,
            "encoder_pretrained": True,
            "label_extensions": None,
        }
        for key, value in optional_defaults.items():
            if not hasattr(args, key):
                setattr(args, key, value)

        # Required nested config sections
        if not hasattr(args, "split_info"):
            raise ValueError("Missing 'split_info' in config.")
        if not hasattr(args, "optimizer"):
            raise ValueError("Missing 'optimizer' in config.")

        args.split_info = _as_namespace(args.split_info)
        args.optimizer = _as_namespace(args.optimizer)
        if getattr(args, "lr_scheduler", None) is not None:
            args.lr_scheduler = _as_namespace(args.lr_scheduler)
            if hasattr(args.lr_scheduler, "params"):
                args.lr_scheduler.params = dict(args.lr_scheduler.params)

        # Device
        args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

        # Seed
        if use_random_seed:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)

        # Encoder pretrained mode
        if not hasattr(args, "encoder_pretrained"):
            args.encoder_pretrained = True
        if cli.no_encoder_pretrained:
            args.encoder_pretrained = False
        elif cli.encoder_pretrained:
            args.encoder_pretrained = True

        # Label extensions may come from config as tuple
        if hasattr(args, "label_extensions"):
            args.label_extensions = _coerce_seq(args.label_extensions)

        # Build paths
        default_root = os.path.join(repo_root, "data", args.dataset)
        if not getattr(args, "dataset_root", None):
            args.dataset_root = default_root
        if not getattr(args, "img_dir", None):
            args.img_dir = os.path.join(args.dataset_root, args.img_folder)
        if not getattr(args, "label_dir", None):
            args.label_dir = os.path.join(args.dataset_root, args.label_folder)

        default_name = splitext(os.path.basename(cli.config))[0]
        args.experim_name = args.experim_name if args.experim_name else default_name

        # Defaults by dataset for model variant (can still be overridden from CLI/config)
        if getattr(args, "model_variant", None) is None:
            args.model_variant = "aachen_baseline" if args.dataset == "aachen" else "uhcs_legacy"

        run_dir = args.output_dir if args.output_dir else os.path.join(repo_root, "checkpoints", args.dataset, args.experim_name)
        self.update_checkpoints_dir(args, run_dir)
        if args.record_path:
            os.makedirs(os.path.dirname(args.record_path), exist_ok=True)

        if verbose:
            self.print_args(args)
        return args

    @staticmethod
    def update_checkpoints_dir(args, checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        args.checkpoints_dir = checkpoints_dir
        args.model_path = args.model_path if args.model_path else os.path.join(checkpoints_dir, "model.pth")
        args.record_path = args.record_path if args.record_path else os.path.join(checkpoints_dir, "train_record.csv")
        args.args_path = os.path.join(checkpoints_dir, "args.yaml")
        args.val_result_path = os.path.join(checkpoints_dir, "val_result.pkl")
        args.test_result_path = os.path.join(checkpoints_dir, "test_result.pkl")
        args.pred_dir = os.path.join(checkpoints_dir, "predictions")
        os.makedirs(args.pred_dir, exist_ok=True)

    @staticmethod
    def print_args(args):
        print(f"Configurations\n{'=' * 50}")
        for k, v in vars(args).items():
            print(k, ":", v)
        print("=" * 50)

    @staticmethod
    def save_args(args, path):
        serializable = {}
        for k, v in vars(args).items():
            if isinstance(v, Namespace):
                serializable[k] = vars(v)
            elif isinstance(v, torch.device):
                serializable[k] = str(v)
            else:
                serializable[k] = v
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(serializable, file, sort_keys=False)


if __name__ == "__main__":
    arg_parser = Arguments()
    _ = arg_parser.parse_args(verbose=True)
