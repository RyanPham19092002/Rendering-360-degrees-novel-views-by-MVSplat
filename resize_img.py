import os
from PIL import Image

folder_path = "/data/Phat/MVSGaussian/examples/route_1"

for step_folder in os.listdir(folder_path):
    for filename in os.listdir(os.path.join(folder_path, step_folder)):
        if filename.endswith('.png') or filename.endswith('.jpeg'):
            file_path = os.path.join(folder_path,step_folder, filename)
            print(file_path)
            img = Image.open(file_path)
            new_img = img.resize((640, 512))
            new_img.save(file_path)
            print(f"Resized image {filename} in {folder_path} successfully.")
