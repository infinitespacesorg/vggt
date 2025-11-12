# python vggt_to_colmap.py --image_dir ./examples/kitchen --output_dir ./outputs/kitchen --conf_threshold 50.0 --mask_sky --mask_black_bg --mask_white_bg --stride 1 --prediction_mode "Depthmap and Camera Branch" --binary

#python vggt_process.py --image_dir ./examples/llff_fern/images/ --output_dir ./examples/llff_fern/sparse --conf_threshold 50.0 --mask_sky --mask_black_bg --mask_white_bg --stride 1 --prediction_mode "Depthmap and Camera Branch" --binary

python vggt_process.py --video_file ./examples/videos/great_wall.mp4 --image_dir ./examples/great_wall/images --output_dir ./examples/great_wall/sparse --fps 1.0 --binary