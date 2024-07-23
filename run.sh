export HYDRA_FULL_ERROR=1
sudo mkdir -p root/VinAI/mvsplat/outputs/local
CUDA_VISIBLE_DEVICES="0" \
python -m src.main +experiment=VinAI checkpointing.load=/data/Phat/mvsplat/data_output_mvsplat/checkpoints/epoch_203-step_110000.ckpt mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=assets/evaluation_index_VinAI_subdata_5_imgs_per_view_nctx2_nf_5000_route1_route2_200x320.json test.compute_scores=true test.target=true