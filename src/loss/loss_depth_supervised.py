from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthSuperviedCfg:
    weight: float


@dataclass
class LossDepthSuperviedCfgWrapper:
    depth_supervised: LossDepthSuperviedCfg


class LossDepthSupervied(Loss[LossDepthSuperviedCfg, LossDepthSuperviedCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        return self.cfg.weight * (delta**2).mean()
