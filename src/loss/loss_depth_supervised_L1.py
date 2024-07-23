import torch
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss

@dataclass
class LossDepthSuperviedCfg:
    weight: float

@dataclass
class LossDepthSuperviedCfgWrapper:
    depth_supervisor: LossDepthSuperviedCfg

class LossDepthSupervied(Loss[LossDepthSuperviedCfg, LossDepthSuperviedCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Lấy giá trị ground truth depth và predicted depth
        y_true = batch["target"]["image_depth"]
        y_pred = prediction.depth
        
        # Đảm bảo y_true và y_pred có cùng kích thước
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        
        # Tránh giá trị zero bằng cách thêm epsilon nhỏ
        epsilon = 1e-6
        log_y_true = torch.log(y_true + epsilon)
        log_y_pred = torch.log(y_pred + epsilon)
        
        # Tính toán sự chênh lệch giữa log của giá trị ground truth và giá trị dự đoán
        diff = log_y_true - log_y_pred
        
        # Tính số lượng điểm dữ liệu
        n = torch.numel(y_true)
        
        # Tính toán scale-invariant log loss
        loss = torch.sum(diff**2) / n - (torch.sum(diff) ** 2) / (n ** 2)
        
        return self.cfg.weight * loss
