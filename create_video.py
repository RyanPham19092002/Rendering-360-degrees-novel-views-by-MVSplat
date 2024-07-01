import os
from pathlib import Path
import re
import glob
from moviepy.editor import ImageSequenceClip

def create_video_from_frames(image_files, output_video, fps=30):
    # Create video clip from image sequence
    clip = ImageSequenceClip(image_files, fps=fps)
    
    # Write the video to file
    clip.write_videofile(output_video, codec='libx264', fps=fps)

def create_video_from_images(step, input_dir, output_dir):
    views = 6
    images_per_view = 20
    
    for root, dirs, files in os.walk(input_dir / f"step_{step}_view_1_00"):
        for d in dirs:
            
            if d.startswith('near_') and 'far_' in d:
    
                image_files = []
                
                for view in range(1, views + 1):
                    view_path = input_dir / f"step_{step}_view_{view}_00" / d / "color"
                    
                    for i in range(images_per_view):
                        image_file = view_path / f"{i:04d}.png"
                        if image_file.exists():
                            image_files.append(str(image_file))
                
                if image_files:
                    video_name = output_dir / f"step_{step}" / d / f"step_{step}.mp4"
                    print(video_name)
                    video_name.parent.mkdir(parents=True, exist_ok=True)
                    create_video_from_frames(image_files, str(video_name))

def get_all_steps(input_dir):
    pattern = re.compile(r'step_(\d+)_view_\d_00')
    step_dirs = glob.glob(str(input_dir / 'step_*_view_*_00'))
    steps = set()
    for dir in step_dirs:
        match = pattern.match(Path(dir).name)
        if match:
            steps.add(match.group(1))
    return sorted(steps, key=int)

def main():
    input_dir = Path("/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/outputs/test/VinAI/view1_non-target_translation")  # Thay thế bằng đường dẫn thư mục input của bạn
    output_dir = Path("/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/outputs/test/video_translation")  # Thay thế bằng đường dẫn thư mục output của bạn
    steps = get_all_steps(input_dir)

    for step in steps:
        create_video_from_images(step, input_dir, output_dir)

if __name__ == "__main__":
    main()
