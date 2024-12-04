python generate.py \
    --oasis-ckpt /data/taiye/Project/open-oasis/models/oasis500m/oasis500m.safetensors \
    --vae-ckpt /data/taiye/Project/open-oasis/models/oasis500m/vit-l-20.safetensors \
    --output-path /data/taiye/Project/open-oasis/outputs/video/video.mp4 \
    --prompt-path sample_data/sample_image_0.png \
    --actions-path data/test_action/1.pt \
    --num-frames 100