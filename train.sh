export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1
sudo mkdir -p /data/Phat/mvsplat/outputs/local
CUDA_VISIBLE_DEVICES="0" python -m src.main +experiment=VinAI data_loader.train.batch_size=2