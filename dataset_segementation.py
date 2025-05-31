from pathlib import Path
import random
from typing import Dict, List, Tuple
from sklearn.model_selection import KFold
import warnings
import cv2
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import rotate
import skimage.io
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm
from torchvision.transforms import ToTensor, Normalize, Compose
import os


def rotated(patch, angle):
    size = patch.shape[:2]
    center = tuple(np.array(size) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(patch, rot_mat, size, flags=cv2.INTER_LINEAR)


def save_image(fname, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(fname, data)


def load_image(path: Path, *, cache: bool) -> np.ndarray:
    cached_path = path.parent / "cache" / (path.stem + ".npy")  # type: Path
    if cache and cached_path.exists():
        return np.load(str(cached_path))
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if path.parent.name == "Train":
        # mask with TrainDotted
        img_dotted = cv2.imread(str(path.parent.parent / "TrainDotted" / path.name))
        img_dotted = cv2.cvtColor(img_dotted, cv2.COLOR_BGR2RGB)
        img[img_dotted.sum(axis=2) == 0, :] = 0
    if cache:
        with cached_path.open("wb") as f:
            np.save(f, img)
    return img


class BaseDataset(Dataset):
    def __init__(
        self,
        img_paths: List[Path],
        coords: pd.DataFrame,
    ):
        self.img_ids = [int(p.name.split(".")[0]) for p in img_paths]
        self.imgs = {
            img_id: load_image(p, cache=True)
            for img_id, p in tqdm.tqdm(
                list(zip(self.img_ids, img_paths)), desc="Images"
            )
        }
        self.coords = coords.loc[self.img_ids].dropna()
        self.coords_by_img_id = {}
        for img_id in self.img_ids:
            try:
                coords = self.coords.loc[[img_id]]
            except KeyError:
                coords = []
            self.coords_by_img_id[img_id] = coords


class BasePatchDataset(BaseDataset):
    def __init__(
        self,
        img_paths: List[Path],
        coords: pd.DataFrame,
        transform,
        size: int,  # patch size
        min_scale: float = 1.0,  # min scale for random scale augmentation
        max_scale: float = 1.0,  # max scale for random scale augmentation
        oversample: float = 0.0,  # probability of oversampling
        deterministic: bool = False,  # if True, the same patch will be sampled for the same index
    ):
        super().__init__(img_paths, coords)
        self.patch_size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.oversample = oversample
        self.transform = transform
        self.deterministic = deterministic

    def __getitem__(self, idx):
        if self.deterministic:
            random.seed(idx)
        while True:
            pp = self.get_patch_points()
            if pp is not None:
                return self.new_x_y(*pp)

    def new_x_y(self, patch, points):
        """Sample (x, y) pair."""
        raise NotImplementedError

    def get_patch_points(self):
        oversample = self.oversample and random.random() < self.oversample
        if oversample:
            item = None
            while item is None or item.name not in self.imgs:
                item = self.coords.iloc[random.randint(0, len(self.coords) - 1)]
            img_id = item.name
        else:
            img_id = random.choice(self.img_ids)
        img = self.imgs[img_id]
        max_y, max_x = img.shape[:2]
        s = self.patch_size
        scale_aug = not (self.min_scale == self.max_scale == 1)
        if scale_aug:
            scale = random.uniform(self.min_scale, self.max_scale)
            s = int(np.round(s / scale))
        else:
            scale = 1
        coords = self.coords_by_img_id[img_id]
        b = int(np.ceil(np.sqrt(2) * s / 2))
        if oversample:
            item = coords.iloc[random.randint(0, len(coords) - 1)]
            x0, y0 = item.col, item.row
            try:
                x = random.randint(max(x0 - s, b), min(x0 + s, max_x - (b + s)))
                y = random.randint(max(y0 - s, b), min(y0 + s, max_y - (b + s)))
            except ValueError:
                oversample = False  # this can happen with large x0 or y0
        if not oversample:
            x = random.randint(b, max_x - (b + s))
            y = random.randint(b, max_y - (b + s))
        patch = img[y - b : y + b + s, x - b : x + b + s]
        angle = random.random() * 360
        patch = rotated(patch, angle)
        patch = patch[b:, b:][:s, :s]
        if (patch == 0).sum() / s**2 > 0.02:
            return None  # masked too much
        if scale_aug:
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        assert patch.shape == (self.patch_size, self.patch_size, 3), patch.shape
        points = []
        if len(coords) > 0:
            for cls, col, row in zip(coords.cls, coords.col, coords.row):
                ix, iy = col - x, row - y
                if (-b <= ix <= b + s) and (-b <= iy <= b + s):
                    p = rotate(Point(ix, iy), -angle, origin=(s // 2, s // 2))
                    points.append((cls, (p.x * scale, p.y * scale)))
        return patch, points

    def __len__(self):
        patch_area = self.patch_size**2
        return int(
            sum(img.shape[0] * img.shape[1] / patch_area for img in self.imgs.values())
        )


class DetectionDataset(BasePatchDataset):
    def __init__(
        self,
        *args,
        bbox_radius: int = 8,
        debug: bool = False,
        downscale=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bbox_radius = bbox_radius
        self.downscale = downscale
        self.debug = debug

    def new_x_y(self, patch, points):
        """Sample (x, bboxes) pair."""
        s_patch = self.patch_size
        bboxes = []
        for cls, (x_coord_in_patch, y_coord_in_patch) in points:
            ix_patch_center, iy_patch_center = int(round(x_coord_in_patch)), int(
                round(y_coord_in_patch)
            )

            xmin = max(0, ix_patch_center - self.bbox_radius)
            ymin = max(0, iy_patch_center - self.bbox_radius)
            xmax = min(s_patch, ix_patch_center + self.bbox_radius)
            ymax = min(s_patch, iy_patch_center + self.bbox_radius)

            # Add valid bounding box: xmin < xmax and ymin < ymax
            if xmin < xmax and ymin < ymax:
                bboxes.append([cls, xmin, ymin, xmax, ymax])

        if self.debug and points:
            # Draw bboxes on patch for debugging
            patch_debug_viz = patch.copy()  # Use the potentially flipped patch
            for bbox_item_viz in bboxes:  # Use the potentially flipped bboxes
                class_id_viz = int(bbox_item_viz[0])
                xmin_viz, ymin_viz, xmax_viz, ymax_viz = map(int, bbox_item_viz[1:])

                # Define some colors for visualization
                colors_viz = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (0, 255, 255),
                    (255, 0, 255),
                ]
                color_viz = colors_viz[class_id_viz % len(colors_viz)]
                cv2.rectangle(
                    patch_debug_viz,
                    (xmin_viz, ymin_viz),
                    (xmax_viz, ymax_viz),
                    color_viz,
                    1,
                )
            save_image("patch_with_boxes.jpg", patch_debug_viz)

        transformed_patch = self.transform(patch)

        if len(bboxes) > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)  # Shape: [N, 5]
        else:
            # If no bounding boxes, return an empty tensor with correct shape
            bboxes_tensor = torch.empty((0, 5), dtype=torch.float32)

        return transformed_patch, bboxes_tensor


def load_coords(data_root: str = None) -> pd.DataFrame:
    coords_path = Path(data_root) / "coords-threeplusone-v0.4.csv"
    return pd.read_csv(coords_path, index_col=0)


def labeled_paths(DATA_ROOT) -> List[Path]:
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/30895
    mismatched = pd.read_csv(str(DATA_ROOT / "MismatchedTrainImages.txt"))
    bad_ids = set(mismatched.train_id)
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31424
    bad_ids.update([941, 200])
    # FIXME - these are valid but have no coords, get them (esp. 912)!
    # https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/discussion/31472#175541
    bad_ids.update([491, 912])
    if os.path.exists(DATA_ROOT / "Train"):
        return [
            p
            for p in DATA_ROOT.joinpath("Train/Train").glob("*.jpg")
            if int(p.stem) not in bad_ids
        ]
    elif os.path.exists(DATA_ROOT / "TrainSmall2"):
        print(
            "Train dataset not found, using TrainSmall2 instead."
        )
        return [
            p
            for p in DATA_ROOT.joinpath("TrainSmall2/Train").glob("*.jpg")
            if int(p.stem) not in bad_ids
        ]
    else:
        raise FileNotFoundError(
            "No Train or TrainSmall2 directory found in the specified DATA_ROOT."
        )


def train_valid_split(
    data_root, coords, stratified=True, fold=1, n_folds=10
) -> Tuple[List[Path], List[Path]]:
    img_paths = labeled_paths(data_root)
    if stratified:
        sorted_ids = coords["cls"].groupby(level=0).count().sort_values().index
        idx_by_id = {img_id: idx for idx, img_id in enumerate(sorted_ids)}
        img_paths.sort(key=lambda p: idx_by_id.get(int(p.stem), len(sorted_ids)))
        train, test = [], []
        for i, p in enumerate(img_paths):
            if i % n_folds == fold - 1:
                test.append(p)
            else:
                train.append(p)
        return train, test
    else:
        img_paths = np.array(sorted(img_paths))
        cv_split = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        img_folds = list(cv_split.split(img_paths))
        train_ids, valid_ids = img_folds[fold - 1]
        return img_paths[train_ids], img_paths[valid_ids]


batch_size = 32
num_workers = 2 if os.name != "nt" else 0
patch_size = 256
oversample = 0.2
bbox_radius = 8
debug = True
n_folds = 10
stratified = True
img_transform = Compose(
    [
        ToTensor(),
        Normalize(mean=[0.44, 0.46, 0.46], std=[0.16, 0.15, 0.15]),
    ]
)

coords = load_coords(data_root="dataset")
train_paths, valid_paths = train_valid_split(
    data_root="dataset", coords=coords, stratified=stratified, n_folds=n_folds
)
train_dataset = DetectionDataset(
    img_paths=train_paths,
    coords=coords,
    size=patch_size,
    transform=img_transform,
    oversample=oversample,
    bbox_radius=bbox_radius,
    debug=debug,
)
valid_dataset = DetectionDataset(
    img_paths=valid_paths,
    coords=coords,
    size=patch_size,
    transform=img_transform,
    oversample=oversample,
    bbox_radius=bbox_radius,
    debug=debug,
    deterministic=True,
)
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    num_workers=num_workers,
    batch_size=batch_size,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    shuffle=False,
    num_workers=num_workers,
    batch_size=batch_size,
)
