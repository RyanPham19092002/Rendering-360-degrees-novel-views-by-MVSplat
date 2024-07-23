import subprocess
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
import argparse
from tqdm import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--json_file", type=str, help="path to the JSON file with camera info")
parser.add_argument("--json_name", type=str, help="path to the name JSON file with camera info")
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)
JSON_FILE = Path(args.json_file)
JSON_NAME = Path(args.json_name)
os.makedirs(str(OUTPUT_DIR), exist_ok=True)
# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)

def build_camera_info_from_json(json_file):
    """Return the camera information from the JSON file"""
    with open(json_file) as f:
        data = json.load(f)

    intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
    scale_factor = 1.0 / 200

    near, far = 0.5, 5000
    # near, far = 425* scale_factor, 25* 192 * scale_factor #original
    for key, transform in data["transform"].items():
        vid = int(key.split('_')[-1])  # Extracting ID from key like "spherical_1"
        world2cam = np.linalg.inv(transform)
        fov = data["fov"]
        width, height = data["img_size"]
        
        fx = width / (2*np.tan(fov * np.pi / 360))
        fy = fx
        cx = width / 2
        cy = height / 2
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        intrinsics[vid] = intrinsic
        world2cams[vid] = world2cam
        cam2worlds[vid] = np.linalg.inv(world2cam)
        near_fars[vid] = [near, far]    
        print("near_fars[vid]-----------------", near_fars[vid])
    return intrinsics, world2cams, cam2worlds, near_fars

def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    """ Extracted from: https://github.com/donydchen/matchnerf/blob/main/configs/dtu_meta/val_all.txt """
    # keys = [
    #     "view_test",
    # ]
    keys = os.listdir(os.path.join(INPUT_IMAGE_DIR,stage))
    print(f"Found {len(keys)} keys.")
    return keys

def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))

def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load images as raw bytes (do not decode)."""
    images_dict = {}
    list_cur_id = os.listdir(example_path)
    for cur_path in list_cur_id:
        # cur_image_name = cur_path.split("/")[-1].split(".")[0]
        cur_id = cur_path.split("/")[-1].split(".")[0].split("_")[-1]
        
        print("cur_id-----", cur_id)
        cur_image_name = cur_path
        print(example_path / cur_image_name)
        img_bin = load_raw(example_path / cur_image_name)
        print("RGB image shape", img_bin.shape)
        images_dict[cur_id] = img_bin

    return images_dict

class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]

class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]
def extract_data(images, example):
    image_keys = set(images.keys())
    sub_example = {
        'url': "",
        'timestamps': [],
        'cameras': []
    }

    for idx, timestep in enumerate(example['timestamps']):
        if str(timestep.item()) in image_keys:
            sub_example['timestamps'].append(timestep)
            sub_example['cameras'].append(example['cameras'][idx])

    # Convert 'cameras' to a list of tensors
    sub_example['timestamps'] = torch.tensor(sub_example['timestamps'], dtype=torch.int64)

    # Chuyển đổi sub_example['cameras'] thành tensor
    sub_example['cameras'] =  torch.tensor(np.stack(sub_example['cameras']), dtype=torch.float32)

    return sub_example
def load_metadata(intrinsics, world2cams) -> Metadata:
    timestamps = []
    cameras = []
    url = ""
    print("intrinsics.items()----------------", intrinsics.items())
    for vid, intr in intrinsics.items():
        print("vid----------", vid)
        timestamps.append(int(vid))

        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = saved_fx
        saved_cx = 0.5
        saved_cy = 0.5
        # saved_fx = fx
        # saved_fy = fy
        # saved_cx = cx
        # saved_cy = cy
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }

if __name__ == "__main__":
    # we only use DTU for testing, not for training
    # for stage in ("train",):
    for stage in ("train", "test"):
        intrinsics, world2cams, cam2worlds, near_fars = build_camera_info_from_json(JSON_FILE)
        # print("--------------Intrinsic----------------")
        # print(intrinsics)
        # print("--------------world2cams----------------")
        # print(world2cams)
        # print("--------------cam2worlds----------------")
        # print(cam2worlds)
        # print("--------------near_fars----------------")
        # print(near_fars)
        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            print("key-------", key)
            # #image gt depth-------------------------------------------------
            image_dir_depth = INPUT_IMAGE_DIR / stage / key / "input_images_raw"
            print("image_dir_depth", image_dir_depth)
            # num_bytes = get_size(image_dir) // 7
            num_bytes_dir = get_size(image_dir_depth)

            # Read images and metadata.
            images_depth = load_images(image_dir_depth)
            # print("images_depth", images_depth)
            # print("images_depth shpae", images_depth.shape)

            #image rgb----------------------------------------------------
            image_dir = INPUT_IMAGE_DIR / stage / key / "input_images"
            print("image_dir", image_dir)
            # num_bytes = get_size(image_dir) // 7
            num_bytes = get_size(image_dir)

            # Read images and metadata.
            images = load_images(image_dir)
            print("num_bytes_dir, num_bytes", num_bytes_dir,num_bytes)
            print("---------------------images----------------")
            print(images)
            example_raw = load_metadata(intrinsics, world2cams)
            print("---------------------example_raw----------------")
            print(example_raw)
            example = extract_data(images, example_raw)
            print("---------------------example----------------")
            print(example)
            # Merge the images into the example.
            print("example[timestamps]----------------", example["timestamps"])
            example["images"] = [
                # images[timestamp.item()] for timestamp in example["timestamps"]
                images[str(timestamp.item())] for timestamp in example["timestamps"]
            ]
            #merge depth
            example["images_depth"] = [
                # images[timestamp.item()] for timestamp in example["timestamps"]
                images_depth[str(timestamp.item())] for timestamp in example["timestamps"]
            ]
            assert len(images) == len(example["timestamps"])

            # Add the key to the example.
            example["key"] = key
            print("example-----------------------", example)
            num_bytes = max(num_bytes, num_bytes_dir)
            # exit(0)
            print(f"Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()
        print("example-----------------------", example)
        
        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        print("index-----------------------", index)
        with (stage_path / JSON_NAME).open("w") as f:
            json.dump(index, f)
