import torch
from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
from nerfstudio.model_components.losses import ScaleAndShiftInvariantLoss

@dataclass
class LossDepthSuperviedCfg:
    weight: float

@dataclass
class LossDepthSuperviedCfgWrapper:
    depth_supervisor: LossDepthSuperviedCfg

class LossDepthSupervied(Loss[LossDepthSuperviedCfg, LossDepthSuperviedCfgWrapper]):
    def __init__(self, cfg: LossDepthSuperviedCfgWrapper):
        super().__init__(cfg)
        self.scale_and_shift_invariant_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=4, reduction_type="batch")
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
        
        mask = (y_true <= 100).float()

        # Rearrange tensors to process them by batch
        y_pred = rearrange(y_pred, "b v h w -> b v 1 h w")
        y_true = rearrange(y_true, "b v h w -> b v 1 h w") / 1000.
        mask = rearrange(mask, "b v h w -> b v 1 h w")

        total_loss = 0.0
        batch_size = y_pred.size(0)
        num_views = y_pred.size(1)
        # Iterate through each batch
        for i in range(batch_size):
            batch_loss = 0.0
            # Iterate through each view in the batch
            for j in range(num_views):
                batch_loss += self.scale_and_shift_invariant_loss(y_pred[i, j], y_true[i, j], mask[i, j])
            # Average loss over the number of views in the batch
            batch_loss /= num_views
            total_loss += batch_loss

        # Average loss over all batches
        loss = total_loss / batch_size
        # torch.cuda.empty_cache()
        # del y_pred, y_true, mask
        # print("loss", loss)
        # exit(0)
        # device = torch.device("cuda:0")
        # device = torch.cuda.set_device(0)
        # loss = loss.to(device)
        return (self.cfg.weight * loss)