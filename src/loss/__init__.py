from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_depth_supervised import LossDepthSupervied, LossDepthSuperviedCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDepthSuperviedCfgWrapper: LossDepthSupervied,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossDepthSuperviedCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    # print(LOSSES[type(cfg)](cfg) for cfg in cfgs)
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
