import os
from os.path import splitext
from typing import List, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


def _load_tiff(path):
    try:
        import tifffile

        return tifffile.imread(path)
    except Exception:
        return np.array(Image.open(path))


class DatasetTemplate(data.Dataset):
    """Template for image/label pair datasets."""

    def __init__(self, img_dir, label_dir, transform, label_mode="raw", label_extensions=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = []
        self.transform = transform
        self.label_mode = label_mode
        self.label_extensions = label_extensions or [".npy", ".tif", ".tiff", ".png"]

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self._get_image(img_name)
        label = self._get_label(img_name)
        img, label = self._transform(img, label)
        return img, label, img_name

    def __len__(self):
        return len(self.img_names)

    def _get_image(self, img_name):
        img_path = f"{self.img_dir}/{img_name}"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _apply_label_mode(self, label):
        if label.ndim > 2:
            label = label[:, :, 0]
        if self.label_mode == "binary_nonzero":
            label = (label > 0).astype(np.int64)
        elif self.label_mode == "raw":
            label = label.astype(np.int64)
        else:
            raise ValueError(f"Unsupported label_mode={self.label_mode}")
        return label

    def _get_label(self, img_name):
        base = img_name.rsplit(".", 1)[0]
        paths = [f"{self.label_dir}/{base}{ext}" for ext in self.label_extensions]

        label = None
        for path in paths:
            if not os.path.exists(path):
                continue
            if path.endswith(".npy"):
                label = np.load(path)
            elif path.endswith(".tif") or path.endswith(".tiff"):
                label = _load_tiff(path)
            elif path.endswith(".png"):
                label = np.array(Image.open(path).convert("L"))
            else:
                continue
            break

        if label is None:
            raise FileNotFoundError(f"No label file found for {img_name}. Tried: {paths}")
        return self._apply_label_mode(label)

    def _transform(self, img, label):
        transformed = self.transform(image=img, mask=label)
        return transformed["image"], transformed["mask"]


class CSVSplitDataset(DatasetTemplate):
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        split_csv: str,
        split_num: Union[int, List[int]],
        transform,
        split_col_name: str = "split",
        reverse: bool = False,
        **kwargs,
    ):
        super().__init__(img_dir, label_dir, transform, **kwargs)
        if isinstance(split_num, (int, np.int64)):
            split_num = [split_num]
        df = pd.read_csv(split_csv)
        if reverse:
            self.img_names = list(df["name"][~df[split_col_name].isin(split_num)])
        else:
            self.img_names = list(df["name"][df[split_col_name].isin(split_num)])


class TextSplitDataset(DatasetTemplate):
    def __init__(self, img_dir, label_dir, split_txt, transform, **kwargs):
        super().__init__(img_dir, label_dir, transform, **kwargs)
        with open(split_txt, "r", encoding="utf-8") as f:
            self.img_names = [line.strip() for line in f.readlines() if line.strip()]


class FolderDataset(DatasetTemplate):
    def __init__(self, img_dir, label_dir, transform, **kwargs):
        super().__init__(img_dir, label_dir, transform, **kwargs)
        self.img_names = [
            name
            for name in os.listdir(self.img_dir)
            if splitext(name)[1].lower() in [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]
        ]
        self.no_label = label_dir is None

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img = self._get_image(img_name)
        if self.no_label:
            h, w, _ = img.shape
            label = -np.ones((h, w), dtype=np.int64)
        else:
            label = self._get_label(img_name)
        img, label = self._transform(img, label)
        return img, label, img_name


class MetadataCSVSplitDataset(DatasetTemplate):
    """Split by metadata file with per-row set column and image_url column."""

    def __init__(
        self,
        img_dir,
        label_dir,
        metadata_csv,
        set_values,
        transform,
        set_col="set",
        metadata_df=None,
        img_names=None,
        **kwargs,
    ):
        super().__init__(img_dir, label_dir, transform, **kwargs)
        self.set_col = set_col
        self.metadata_df = metadata_df
        self.img_names = img_names or []

        if self.metadata_df is None:
            self.metadata_df = pd.read_csv(metadata_csv)
            if "image_url" not in self.metadata_df.columns:
                raise ValueError(f"Column 'image_url' not found in {metadata_csv}")
            self.metadata_df = self.metadata_df[self.metadata_df[set_col].isin(set_values)]
            self.img_names = [name.strip() for name in self.metadata_df["image_url"].dropna().tolist()]
            if len(self.img_names) == 0:
                raise ValueError(f"No images found for sets {set_values} in {metadata_csv}")

        self.set_map = {row["image_url"]: row[set_col] for _, row in self.metadata_df.iterrows()}

    def _get_image(self, img_name):
        set_name = self.set_map.get(img_name, "train")
        sub_img_dir = os.path.join(os.path.dirname(self.img_dir), set_name, os.path.basename(self.img_dir))
        img_path = f"{sub_img_dir}/{img_name}"
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _get_label(self, img_name):
        set_name = self.set_map.get(img_name, "train")
        sub_label_dir = os.path.join(os.path.dirname(self.label_dir), set_name, os.path.basename(self.label_dir))
        base = img_name.rsplit(".", 1)[0]
        paths = [f"{sub_label_dir}/{base}{ext}" for ext in self.label_extensions]

        label = None
        for path in paths:
            if not os.path.exists(path):
                continue
            if path.endswith(".npy"):
                label = np.load(path)
            elif path.endswith(".tif") or path.endswith(".tiff"):
                label = _load_tiff(path)
            elif path.endswith(".png"):
                label = np.array(Image.open(path).convert("L"))
            break

        if label is None:
            raise FileNotFoundError(f"No label found for {img_name} in set={set_name}. Tried: {paths}")
        return self._apply_label_mode(label)


def get_list_of_ops(augs, library):
    if augs is None:
        return []
    ops = []
    if isinstance(augs, list):
        for func_name in augs:
            func = getattr(library, func_name)
            ops.append(func(p=0.5))
    elif isinstance(augs, dict):
        for func_name, kwargs in augs.items():
            func = getattr(library, func_name)
            ops.append(func(**kwargs))
    else:
        raise ValueError(f"Unsupported augmentation config type: {type(augs)}")
    return ops


def get_transform(args, is_train):
    train_size = tuple(getattr(args, "train_size", [512, 512]))
    eval_size = tuple(getattr(args, "eval_size", train_size))
    mean = getattr(args, "mean", [0.485, 0.456, 0.406])
    std = getattr(args, "std", [0.229, 0.224, 0.225])

    if is_train:
        ops = []
        if getattr(args, "use_random_crop", False):
            ops.append(A.RandomCrop(*train_size))
        else:
            ops.append(A.Resize(*train_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST))
        ops.extend(get_list_of_ops(getattr(args, "augmentations", None), A))
        ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
        return A.Compose(ops)
    return A.Compose([A.Resize(*eval_size), A.Normalize(mean=mean, std=std), ToTensorV2()])


def _dataset_label_mode(args):
    if hasattr(args, "label_mode") and args.label_mode:
        return args.label_mode
    # Sensible defaults
    return "binary_nonzero" if args.dataset == "aachen" else "raw"


def _dataset_label_extensions(args):
    if hasattr(args, "label_extensions") and args.label_extensions:
        return list(args.label_extensions)
    return [".npy", ".tif", ".tiff", ".png"]


def _loader_kwargs(args, train):
    if train:
        return dict(
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(getattr(args, "train_num_workers", 4)),
            pin_memory=bool(getattr(args, "pin_memory", True)),
            drop_last=bool(getattr(args, "drop_last", True)),
        )
    return dict(
        batch_size=1,
        shuffle=False,
        num_workers=int(getattr(args, "eval_num_workers", 2)),
        pin_memory=bool(getattr(args, "pin_memory", True)),
    )


def get_dataloaders(args):
    transform_train = get_transform(args, is_train=True)
    transform_eval = get_transform(args, is_train=False)
    s_info = args.split_info

    common_kwargs = dict(
        label_mode=_dataset_label_mode(args),
        label_extensions=_dataset_label_extensions(args),
    )

    if s_info.type == "CSVSplit":
        split_file_path = f"{args.dataset_root}/splits/{s_info.split_file}"
        train_split_num = [s_info.val_split_num, s_info.test_split_num] if s_info.train_reverse else s_info.train_split_num
        train_set = CSVSplitDataset(
            args.img_dir,
            args.label_dir,
            split_csv=split_file_path,
            split_num=train_split_num,
            transform=transform_train,
            split_col_name=s_info.split_col_name,
            reverse=s_info.train_reverse,
            **common_kwargs,
        )
        val_set = CSVSplitDataset(
            args.img_dir,
            args.label_dir,
            split_csv=split_file_path,
            split_num=s_info.val_split_num,
            transform=transform_eval,
            split_col_name=s_info.split_col_name,
            **common_kwargs,
        )
        if getattr(s_info, "test_type", "validation") == "validation":
            test_set = val_set
        elif s_info.test_type == "CSVSplit":
            test_set = CSVSplitDataset(
                args.img_dir,
                args.label_dir,
                split_csv=split_file_path,
                split_num=s_info.test_split_num,
                transform=transform_eval,
                split_col_name=s_info.split_col_name,
                **common_kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported test_type={s_info.test_type}")

    elif s_info.type == "TextSplit":
        train_split_file_path = f"{args.dataset_root}/splits/{s_info.train_split_file}"
        val_split_file_path = f"{args.dataset_root}/splits/{s_info.val_split_file}"
        train_set = TextSplitDataset(args.img_dir, args.label_dir, train_split_file_path, transform_train, **common_kwargs)
        val_set = TextSplitDataset(args.img_dir, args.label_dir, val_split_file_path, transform_eval, **common_kwargs)

        if getattr(s_info, "test_type", "validation") == "validation":
            test_set = val_set
        elif s_info.test_type == "TextSplit":
            test_split_file_path = f"{args.dataset_root}/splits/{s_info.test_split_file}"
            test_set = TextSplitDataset(
                args.img_dir,
                args.label_dir,
                test_split_file_path,
                transform_eval,
                **common_kwargs,
            )
        elif s_info.test_type == "folder":
            test_set = FolderDataset(s_info.test_img_dir, s_info.test_label_dir, transform_eval, **common_kwargs)
        else:
            raise NotImplementedError(f"Unsupported test_type={s_info.test_type}")

    elif s_info.type == "MetadataCSV":
        metadata_path = f"{args.dataset_root}/{s_info.metadata_csv}"
        full_metadata = pd.read_csv(metadata_path)
        if "image_url" not in full_metadata.columns:
            raise ValueError(f"Column 'image_url' not found in {metadata_path}")

        train_sets = list(getattr(s_info, "train_sets", ["train"]))
        val_sets = list(getattr(s_info, "val_sets", []))
        test_sets = list(getattr(s_info, "test_sets", ["test"]))
        set_col = getattr(s_info, "set_col", "set")

        train_df = full_metadata[full_metadata[set_col].isin(train_sets)].copy().reset_index(drop=True)
        if len(train_df) == 0:
            raise ValueError(f"No train rows found for {train_sets} in {metadata_path}")

        if val_sets:
            val_df = full_metadata[full_metadata[set_col].isin(val_sets)].copy().reset_index(drop=True)
        else:
            val_ratio = float(getattr(s_info, "val_split_ratio", 0.1))
            n_val = max(1, int(len(train_df) * val_ratio))
            rng = np.random.default_rng(int(getattr(args, "seed", 42)))
            val_idx = rng.choice(len(train_df), size=n_val, replace=False)
            train_idx = np.setdiff1d(np.arange(len(train_df)), val_idx)
            val_df = train_df.iloc[val_idx].reset_index(drop=True)
            train_df = train_df.iloc[train_idx].reset_index(drop=True)

        test_df = full_metadata[full_metadata[set_col].isin(test_sets)].copy().reset_index(drop=True)

        train_set = MetadataCSVSplitDataset(
            img_dir=f"{args.dataset_root}/{args.img_folder}",
            label_dir=f"{args.dataset_root}/{args.label_folder}",
            metadata_csv=metadata_path,
            set_values=train_sets,
            transform=transform_train,
            set_col=set_col,
            metadata_df=train_df,
            img_names=train_df["image_url"].tolist(),
            **common_kwargs,
        )
        val_set = MetadataCSVSplitDataset(
            img_dir=f"{args.dataset_root}/{args.img_folder}",
            label_dir=f"{args.dataset_root}/{args.label_folder}",
            metadata_csv=metadata_path,
            set_values=val_sets if val_sets else ["val"],
            transform=transform_eval,
            set_col=set_col,
            metadata_df=val_df,
            img_names=val_df["image_url"].tolist(),
            **common_kwargs,
        )
        if len(test_df) == 0:
            test_set = val_set
        else:
            test_set = MetadataCSVSplitDataset(
                img_dir=f"{args.dataset_root}/{args.img_folder}",
                label_dir=f"{args.dataset_root}/{args.label_folder}",
                metadata_csv=metadata_path,
                set_values=test_sets,
                transform=transform_eval,
                set_col=set_col,
                metadata_df=test_df,
                img_names=test_df["image_url"].tolist(),
                **common_kwargs,
            )
    else:
        raise NotImplementedError(f"Unsupported split type={s_info.type}")

    if int(getattr(args, "train_repeat", 1)) > 1:
        train_set = data.ConcatDataset([train_set] * int(args.train_repeat))

    train_loader = DataLoader(train_set, **_loader_kwargs(args, train=True))
    val_loader = DataLoader(val_set, **_loader_kwargs(args, train=False))
    test_loader = DataLoader(test_set, **_loader_kwargs(args, train=False))
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from args import Arguments

    parser = Arguments()
    args = parser.parse_args(verbose=True, use_random_seed=False)
    _ = get_dataloaders(args)
