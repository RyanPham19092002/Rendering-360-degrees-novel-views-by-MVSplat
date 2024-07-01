import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetVinAICfg(DatasetCfgCommon):
    name: Literal["VinAI"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = 0.5
    far: float = 5000.
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetVinAI(IterableDataset):
    cfg: DatasetVinAICfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.5
    far: float = 5000
    # near: float = 0.1
    # far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetVinAICfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            print("over to scec", self.cfg.overfit_to_scene)
            print("index",self.index)
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            print("-----------------chunk_path------------", chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)
            print("self.cfg.shuffle_val--------------------", self.cfg.shuffle_val)
            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            print("times_per_scene-------------", times_per_scene)
            print("len(chunk)---------------------", len(chunk))
            for run_idx in range(int(times_per_scene * len(chunk))):
                print("run_idx---------------------", run_idx)
                example = chunk[run_idx // times_per_scene]
                

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                    #scene = f"{example['key']}"
                    # print("scene--------------", scene)
                else:
                    scene = example["key"]
                # print("scene--------------", scene)
                # print("extrinsics--------------", extrinsics)
                # print("intrinsics--------------", intrinsics)
                # context_indices, target_indices = self.view_sampler.sample(
                #         scene,
                #         extrinsics,
                #         intrinsics,
                #     )
                print("context_indices-----------------", context_indices)
                # print("target_indices-----------------", target_indices)
                try:
                    # context_indices, target_indices = self.view_sampler.sample(
                    context_indices, _ = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # print("context_indices-----------------", context_indices)
                    # print("target_indices-----------------", target_indices)
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                # target_images = [
                #     example["images"][index.item()] for index in target_indices
                # ]
                # target_images = self.convert_images(target_images)
                # print("context_indices-----------------", context_indices)
                # print("target_indices-----------------", target_indices)
                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, 360, 640)
                # target_image_invalid = target_images.shape[1:] != (3, 360, 640)
                # if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                #     print(
                #         f"Skipped bad example {example['key']}. Context shape was "
                #         f"{context_images.shape} and target shape was "
                #         f"{target_images.shape}."
                #     )
                #     continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    # "target": {
                    #     "extrinsics": extrinsics[target_indices],
                    #     "intrinsics": intrinsics[target_indices],
                    #     "image": target_images,
                    #     "near": self.get_bound("near", len(target_indices)) / nf_scale,
                    #     "far": self.get_bound("far", len(target_indices)) / nf_scale,
                    #     "index": target_indices,
                    # },
                    "scene": scene,
                }
                # print("example---------------", example)
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy
        # print("fx,fy,cx,cy----------------", fx,fy,cx,cy)
        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        #print("intrinsics------------", intrinsics)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
        # data_stage = "test"
            for root in self.cfg.roots:
                # Load the root's index.
                # print(root / data_stage / "index_VinAI_6_input_full_views_nf_5000.json")
                with (root / data_stage / "evaluation_index_VinAI_subdata_120_imgs_view_nctx2_nf_5000.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}
                print("index", index)
                # The constituent datasets should have unique keys.
                print("merged_index keys:", merged_index.keys())
                print("index keys:", index.keys())
                print("Common keys:", set(merged_index.keys()) & set(index.keys()))
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
