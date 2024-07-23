from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
import cv2
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
# import open3d as o3d
from PIL import Image
import os

from ..visualization.vis_depth import viz_depth_tensor
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .interpolated_poses import get_interpolated_poses
# from ...create_video import get_all_steps, create_video_from_images
from src.geometry.projection import homogenize_points, project
# Function to remove extra points or colors to match the number of points and colors
def remove_extra(data, target_length):
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        ratio = len(data) / target_length
        indices = [int(i * ratio) for i in range(target_length)]
        return [data[i] for i in indices]
    return data

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    target : bool
    translation_pose: float


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        # print("batch depth img", batch["target"]["image_depth"])
        # print("batch depth img shape", batch["target"]["image_depth"].shape)
        _, _, _, h, w = batch["target"]["image"].shape
        # print("h,w on train step", h,w)
        print("On train step--------------------------------------")
        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]
        # print("target_gt shape", target_gt.shape)
        # print("output.color shape", output.color.shape)
        target_depth_images = batch["target"]["image_depth"]
        # depth_map_pred = output.depth
        # print("depth_map_pred", depth_map_pred)
        # print("depth_map_pred shape", depth_map_pred.shape)
 
        
        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            # print(f"--------------------------{loss_fn}-----------------------------")
            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
            # print("loss", loss)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            # print("batch train-----------------", batch)
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"target = {batch['target']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss
 
    def test_step(self, batch, batch_idx):
        #--------------------------------view non target---------------------------------#
        
        if not self.test_cfg.target:
            print("batch", batch)
            def save_interpolated_pose_to_json(index_pose, interpolated_pose, output_file):
                # Open the file in append mode and write the interpolated pose
                with open(output_file, 'a') as f:
                    key = f"input_camera_{index_pose}"
                    json.dump({key: interpolated_pose.tolist()}, f)
                    f.write('\n')

            FIGURE_WIDTH = 500
            MARGIN = 4
            GAUSSIAN_TRIM = 8
            LINE_WIDTH = 1.8
            LINE_COLOR = [255, 0, 0]
            POINT_DENSITY = 0.5
            # print(intrinsic_matrix_raw[0,0,0])
            batch: BatchedExample = self.data_shim(batch)
            #b, v, _, h, w = batch["target"]["image"].shape
            b, v, _, h, w = batch["context"]["image"].shape
            print("h, w on test step", h, w)
            assert b == 1
            print("On test step--------------------------------------")
            # Render Gaussians.
            with self.benchmarker.time("encoder"):
                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )
                visualization_dump = {}
                gaussians_forward = self.encoder.forward(
                    batch["context"], False, visualization_dump=visualization_dump
                )
                
                # print("gaussians_forward mean-----------------")
                # print(gaussians_forward.means)
                # print("gaussians_forward harmonics-----------------")
                # print(gaussians_forward.harmonics)
                # print("gaussians_forward opacities-----------------")
                # print(gaussians_forward.opacities)
            # print('batch["context"]["extrinsics"]--------------', batch["context"]["extrinsics"])
            # print('batch["context"]["extrinsics"] type--------------', type(batch["context"]["extrinsics"]))
            # print('batch["target"]["extrinsics"] shape--------------', batch["context"]["extrinsics"].shape)
            # print('batch["context"]["intrinsics"]shape---------------------', batch["context"]["intrinsics"].shape)
            # print('batch["context"]["intrinsics"]---------------------', batch["context"]["intrinsics"])
            intrinsic_matrix = batch["context"]["intrinsics"][0][0].unsqueeze(0).unsqueeze(0)
            # print('intrinsic_matrix---------------------', intrinsic_matrix.shape)
            # print('intrinsic_matrix---------------------', intrinsic_matrix)
            # print('batch["context"]["near"]shape---------------------', batch["context"]["near"].shape)
            # print('batch["context"]["far"] shape---------------------', batch["context"]["far"].shape)
            # print('batch["context"]["near"]---------------------', batch["context"]["near"])
            # print('batch["context"]["far"]---------------------', batch["context"]["far"])
            near_value = batch["context"]["near"][:, 0].unsqueeze(dim=1)
            far_value = batch["context"]["far"][:, 0].unsqueeze(dim=1)
            near = float(near_value.item())
            far = float(far_value.item())           

            pose_a = batch["context"]["extrinsics"][0][0]
            pose_b = batch["context"]["extrinsics"][0][1]
            pose_a = pose_a.cpu().numpy()
            pose_b = pose_b.cpu().numpy()
            # print("---------------before-------------------")
            # print("pose_A", pose_a.shape)
            # print("pose_A", pose_a)
            # print("-----------------------------------")
            # print("pose_b", pose_b.shape)
            # print("pose_b", pose_b)
            print("self.test_cfg.translation_pose", self.test_cfg.translation_pose)
            #raw
            # pose_a[:2, 3] -= self.test_cfg.translation_pose             
            # pose_b[:2, 3] -= self.test_cfg.translation_pose   
            #3rd view
            pose_a[:1, 3] -= self.test_cfg.translation_pose             
            pose_b[:1, 3] -= self.test_cfg.translation_pose   
            pose_a[2, 3] = 1.25        
            pose_b[2, 3] = 1.25
            # print("---------------after-------------------")
            # print("pose_A", pose_a.shape)
            # print("pose_A", pose_a)
            # print("-----------------------------------")
            # print("pose_b", pose_b.shape)
            # print("pose_b", pose_b)
            # exit(0)
            step = "3view_test_route1_110k"
            if self.test_cfg.translation_pose  != 0:
                type_folder = f"_non-target_translation_{step}_step"
            else:
                type_folder = f"_non-target_{step}_step"
            interpolated_poses = get_interpolated_poses(pose_a, pose_b, 20)
            # print("interpolated_poses--------------", type(interpolated_poses))
            points_list_depth = []
            colors_list = []
            for index_pose, interpolated_pose in enumerate(interpolated_poses):
                (scene,) = batch["scene"]
                name = get_cfg()["wandb"]["name"]
                path = self.test_cfg.output_path / Path(name + type_folder)
                folder_path = path / scene / f"near_{near}_far_{far}/"
                os.makedirs(folder_path, exist_ok=True)
                output_file = path / scene / f"near_{near}_far_{far}/extrinsic_pose_not_inv.json"
                # print("index_pose------------------------", index_pose)
                # print("-------------before-------------")
                # print('interpolated_pose shape--------------', interpolated_pose.shape)
                # print('interpolated_pose--------------', type(interpolated_pose))
                last_row = np.array([0, 0, 0, 1], dtype=np.float32)
                interpolated_pose = np.vstack([interpolated_pose, last_row])
                save_interpolated_pose_to_json(index_pose, interpolated_pose, output_file)
                interpolated_pose = torch.tensor(interpolated_pose).unsqueeze(0).unsqueeze(0)
                # save_interpolated_pose_to_json(index_pose, interpolated_pose, output_file)
                # print("-------------after-------------")
                # print('interpolated_pose shape--------------',interpolated_pose.shape)
                # print('interpolated_pose--------------', interpolated_pose)
                # print('type interpolated_pose--------------', type(interpolated_pose))
                with self.benchmarker.time("decoder", num_calls=v):

                    # print('batch["context"]["extrinsics"]--------------', batch["context"]["extrinsics"])
                    device = intrinsic_matrix.device
                    # print(device)
                    # Chuyển interpolated_pose lên cùng thiết bị
                    interpolated_pose = interpolated_pose.to(device=device, dtype=torch.float32)
                    # print("interpolated_pose--------------")
                    # print(interpolated_pose.shape)
                    # print(interpolated_pose)
                    depth_mode="depth"
                    output = self.decoder.forward(
                        gaussians,
                        # batch["target"]["extrinsics"],
                        # batch["target"]["intrinsics"],
                        # batch["target"]["near"],
                        # batch["target"]["far"],
                        interpolated_pose,
                        intrinsic_matrix,
                        near_value,
                        far_value,
                        (h, w),
                        depth_mode=depth_mode,
                    )

                
                images_prob = output.color[0]
                # images_depth_prob = output.depth[0]
                images_depth_prob = output.depth.cpu().detach()
                save_path = path / scene / f"near_{near}_far_{far}/{depth_mode}"
                os.makedirs(save_path, exist_ok=True)
                save_path_file_ply = path / scene / f"near_{near}_far_{far}/file_ply"
                os.makedirs(save_path_file_ply, exist_ok=True)
                def convert_depth_map(depth_map, intrinsic_old, intrinsic_new):

                    if isinstance(depth_map, torch.Tensor):
                        depth_map = depth_map.cpu().numpy()

                    if isinstance(intrinsic_old, torch.Tensor):
                        intrinsic_old = intrinsic_old.cpu().numpy()

                    if isinstance(intrinsic_new, torch.Tensor):
                        intrinsic_new = intrinsic_new.cpu().numpy()

                    H, W = depth_map.shape
        
                    # Tính toán ma trận biến đổi từ intrinsic_old sang intrinsic_new
                    transform_matrix = intrinsic_new @ np.linalg.inv(intrinsic_old)
                    
                    # Tạo meshgrid cho tọa độ pixel
                    y, x = np.meshgrid(np.arange(W), np.arange(H))
                    
                    # Chuyển về dạng flattened
                    x = x.flatten()
                    y = y.flatten()
                    depth = depth_map.flatten()
                    
                    # Tạo ma trận [x, y, 1]
                    ones = np.ones_like(x)
                    coordinates = np.vstack((x, y, depth))
                    
                    # Áp dụng biến đổi để tính toán tọa độ mới
                    new_coordinates = transform_matrix @ coordinates

                    print("new_coordinates", new_coordinates)
                    # Lấy ra tọa độ mới x, y và chia cho z để chuẩn hóa
                    x_new = (new_coordinates[0][0, :] / new_coordinates[0][2, :]).reshape(H, W).astype(np.float32)
                    y_new = (new_coordinates[0][1, :] / new_coordinates[0][2, :]).reshape(H, W).astype(np.float32)
                    # x_new_normalized = (x_new / x_new.max() * (W - 1)).round()
                    # y_new_normalized = (y_new / y_new.max() * (H - 1)).round()
                    print("x_new, y_new", x_new, y_new)
                    # Áp dụng remap để chuyển đổi depth map
                    new_depth_map = cv2.remap(depth_map.astype(np.float32), x_new, y_new, interpolation=cv2.INTER_LINEAR)
                    max_value = np.max(new_depth_map)

                    print("Maximum value in new_depth_map:", max_value)
                    return new_depth_map
                #point cloud--------------------------------------------------------------------------------
                # print("images_prob", images_prob)
                # print("images_prob shape", images_prob.shape)    
                # print("images_depth_prob", images_depth_prob)
                # print("images_depth_prob shape", images_depth_prob.shape)
                #rgb_gt = batch["target"]["image"][0]
                print("scene-----------------", scene)
                # Save images.
                if self.test_cfg.save_image:
                    name = [index_pose]
                    # for index, color in zip(batch["target"]["index"][0], images_prob):
                    for index, color in zip(name, images_prob):
                        # print("color--------------")
                        # print(color)
                        save_image(color, path / scene / f"near_{near}_far_{far}/color/{index:0>4}.png")
                    for index, depth in zip(name, images_depth_prob):
                        for v_idx in range(images_depth_prob.shape[1]):
                            print("images_depth_prob[0, v_idx] shape", images_depth_prob[0, v_idx].shape)
                            # print("images_depth_prob[0, v_idx]", images_depth_prob[0, v_idx])
                            vis_depth = viz_depth_tensor(
                               1 / images_depth_prob[0, v_idx], return_numpy=True
                            ) 
                            print(f"{save_path}/{index:0>4}.png")
                            Image.fromarray(vis_depth).save(f"{save_path}/{index:0>4}.png")
            print("Done extract images!!")
    
        #--------------------------------------------------------------------------------#
        #--------------------------------view target-------------------------------------#
        else:

            batch: BatchedExample = self.data_shim(batch)
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1

            # Render Gaussians.
            with self.benchmarker.time("encoder"):
                gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )
            with self.benchmarker.time("decoder", num_calls=v):
                depth_mode="depth"
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=depth_mode,
                )
            near_value = batch["context"]["near"][:, 0].unsqueeze(dim=1)
            far_value = batch["context"]["far"][:, 0].unsqueeze(dim=1)
            near = float(near_value.item())
            far = float(far_value.item()) 
            (scene,) = batch["scene"]
            name = get_cfg()["wandb"]["name"]
            step_folder = "route2_110k"
            path = self.test_cfg.output_path / Path(name + f'_target_{step_folder}_step')
            folder_path = path / scene / f"near_{near}_far_{far}/"
            os.makedirs(folder_path, exist_ok=True)
            images_prob = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            depth_gt = batch["target"]["image_depth"].cpu().detach()
            print("depth_gt shape", depth_gt.shape)
            batch_size, num_views, height, width = depth_gt.shape
            for b in range(batch_size):
                for v in range(num_views):
                    # Lấy depth map
                    depth_map = depth_gt[b, v, :, :]

                    # Tạo một mask với các giá trị depth <= 100
                    mask = (depth_map <= 100).numpy().astype(np.uint8)

                    # Tạo tên file cho từng depth map
                    filename = f"binary_mask_batch{b}_view{v}.png"

                    # Lưu mask dưới dạng ảnh
                    cv2.imwrite(filename, mask * 255)
            exit(0)
            #---------------------------------------------------------------------
            
            images_depth_prob = output.depth.cpu().detach()
            # print("images_depth_prob shape", images_depth_prob.shape)
            save_path = path / scene / f"near_{near}_far_{far}/{depth_mode}"
            os.makedirs(save_path, exist_ok=True)
            #render depth target -----------------------------------------------------------------------
            rgb_softmax = output.color[0]
            images_depth_prob = output.depth.cpu().detach()
            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(
                ("val",), (rgb_softmax,)
            ):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)
            
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Softmax)"),
            )
            self.logger.log_image(
                str(os.path.join("comparison", Path(name + f'_target_{step_folder}_step')/ scene / f"near_{near}_far_{far}/")),
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
            # render depthmap gt
            vis_depth_gt_list = []
            for v_idx in range(depth_gt.shape[1]):
                # print("depth_gt[0, v_idx] shape", depth_gt[0, v_idx].shape)
                vis_depth_gt = viz_depth_tensor(
                    1.0 / depth_gt[0, v_idx], return_numpy=True
                ) 

                vis_depth_gt_list.append(torch.from_numpy(vis_depth_gt/255.))
            vis_depth_gt_tensor = torch.stack(vis_depth_gt_list).permute(0, 3, 1, 2)
            # render depthmap validation
            vis_depth_list = []
            for v_idx in range(images_depth_prob.shape[1]):
                # print("images_depth_prob[0, v_idx] shape", images_depth_prob[0, v_idx].shape)
                vis_depth = viz_depth_tensor(
                    1.0 / images_depth_prob[0, v_idx], return_numpy=True
                ) 

                vis_depth_list.append(torch.from_numpy(vis_depth/255.))
            vis_depth_tensor = torch.stack(vis_depth_list).permute(0, 3, 1, 2)
            

            depth = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*vis_depth_gt_tensor), "Depth (Ground Truth)"),
                add_label(vcat(*vis_depth_tensor), "Predict depth"),
            )
            self.logger.log_image(
                str(os.path.join("depth", Path(name + f'_target_{step_folder}_step')/ scene / f"near_{near}_far_{far}/")),
                # "depth",
                [prep_image(add_border(depth))],
                step=self.global_step,
                caption=batch["scene"],
            )
            # Render projections and construct projection image.
            projections = hcat(*render_projections(
                                    gaussians,
                                    256,
                                    extra_label="(Softmax)",
                                )[0])
            self.logger.log_image(
                str(os.path.join("projection", Path(name + f'_target_{step_folder}_step')/ scene / f"near_{near}_far_{far}/")),
                # "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

            # Draw cameras.
            cameras = hcat(*render_cameras(batch, 256))
            self.logger.log_image(
                str(os.path.join("cameras", Path(name + f'_target_{step_folder}_step')/ scene / f"near_{near}_far_{far}/")),
                # "cameras", 
                [prep_image(add_border(cameras))], step=self.global_step
            )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)
            # Save images.
            if self.test_cfg.save_image:
                # print('batch["target"]["index"][0]', batch["target"]["index"][0])
                for index, color in zip(batch["target"]["index"][0], images_prob):
                    save_image(color, path / scene / f"near_{near}_far_{far}/color/{index:0>4}.png")
                # name = [1,2,3]
                # print('batch["target"]["index"][0]', batch["target"]["index"][0])
                
                # for index, depth in zip(name, images_depth_prob):
                for v_idx in range(images_depth_prob.shape[1]):
                    # print("images_depth_prob[0, v_idx] shape", images_depth_prob[0, v_idx].cpu().numpy().shape)
                    # print("images_depth_prob[0, v_idx]", images_depth_prob[0, v_idx].cpu().numpy())
                    
                    vis_depth = viz_depth_tensor(
                        1.0 / images_depth_prob[0, v_idx], return_numpy=True
                    ) 
                    print(f"{save_path}/{(v_idx+1):0>4}.png")
                    Image.fromarray(vis_depth).save(f"{save_path}/{(v_idx+1):0>4}.png")
            # compute scores
            if self.test_cfg.compute_scores:
                if batch_idx < self.test_cfg.eval_time_skip_steps:
                    self.time_skip_steps_dict["encoder"] += 1
                    self.time_skip_steps_dict["decoder"] += v
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )
        
        # if self.test_cfg.save_video:
        #         # input_dir =
        #         print("Done extract video!!")
        
    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / Path(name + '_target')
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        # print("batch validation------", batch)
        batch: BatchedExample = self.data_shim(batch)
        print("In validation step-------------------------------------")
        if self.global_rank == 0:
            print(
                # batch['context'],
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()};"
                f"target = {batch['target']['index'].tolist()};"
            )
            # print("batch------", batch)
        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],            
            (h, w),
            depth_mode="depth"
        )
        rgb_softmax = output_softmax.color[0]
        images_depth_prob = output_softmax.depth.cpu().detach()
        # print("images_depth_prob shape", images_depth_prob.shape)
        # print("images_depth_prob", images_depth_prob)
        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        # print("rgb_gt shape", rgb_gt.shape)
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)
        
        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
        # render depthmap validation
        
        vis_depth_list = []
        for v_idx in range(images_depth_prob.shape[1]):
            # print("images_depth_prob[0, v_idx] shape" ,images_depth_prob[0, v_idx].shape)
            vis_depth = viz_depth_tensor(
                
                1.0 / images_depth_prob[0, v_idx], return_numpy=True
            ) 

            vis_depth_list.append(torch.from_numpy(vis_depth/255.))
        vis_depth_tensor = torch.stack(vis_depth_list).permute(0, 3, 1, 2)
        # print("vis_depth_tensor shape",vis_depth_tensor.shape)
        # print("vis_depth_tensor",vis_depth_tensor)
        depth_gt = batch["target"]["image_depth"].cpu().detach()
        # print("depth_gt shape", depth_gt.shape)
        # print("depth_gt", depth_gt)
        vis_depth_gt_list = []
        for v_idx_depth in range(depth_gt.shape[1]):
            # print("depth_gt[0, v_idx] shape" ,depth_gt[0, v_idx_depth].shape)
            # print("depth_gt[0, v_idx]", depth_gt[0, v_idx_depth])
            # print("1.0 / (depth_gt[0, v_idx_depth])", 1.0 / depth_gt[0, v_idx_depth])
            vis_depth_gt = viz_depth_tensor(
                1.0 / depth_gt[0, v_idx_depth], return_numpy=True
            ) 
            # print(f"----------{v_idx_depth}----------------")
            # print(vis_depth_gt/255.)
            vis_depth_gt_list.append(torch.from_numpy(vis_depth_gt/255.))

        vis_depth_gt_tensor = torch.stack(vis_depth_gt_list).permute(0, 3, 1, 2)
        # print("vis_depth_gt_tensor shape",vis_depth_gt_tensor.shape)
        # print("vis_depth_gt_tensor",vis_depth_gt_tensor)
        
        # depth_gt = batch["target"]["image_depth"][0]/1000.
        # depth_gt = depth_gt.unsqueeze(1)
        depth = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*vis_depth_gt_tensor), "Depth (Ground Truth)"),
            add_label(vcat(*vis_depth_tensor), "Predict depth"),
        )
        self.logger.log_image(
            "depth",
            [prep_image(add_border(depth))],
            step=self.global_step,
            caption=batch["scene"],
        )
        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    # if v == 2
                    # else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
