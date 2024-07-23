import os
from PIL import Image


folder_path = "/data/Phat/VinAI/subdata_mvsgaussian_depth_route1_route2_p2/train"

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpeg'):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        print(f"Image {filename} has size {img.size}.")
