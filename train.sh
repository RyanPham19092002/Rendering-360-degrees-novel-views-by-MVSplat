sudo mkdir -p root/VinAI/mvsplat/outputs/local
CUDA_VISIBLE_DEVICES="0" python -m src.main +experiment=VinAI data_loader.train.batch_size=5