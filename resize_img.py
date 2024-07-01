import os
from PIL import Image

folder_path = "/home/ubuntu/Workspace/phat-intern-dev/VinAI/subdata_remain/train"

for step_folder in os.listdir(folder_path):
    for filename in os.listdir(os.path.join(folder_path, step_folder)):
        if filename.endswith('.png') or filename.endswith('.jpeg'):
            file_path = os.path.join(folder_path,step_folder, filename)
            print(file_path)
            img = Image.open(file_path)
            new_img = img.resize((320, 200))
            new_img.save(file_path)
            print(f"Resized image {filename} in {folder_path} successfully.")
