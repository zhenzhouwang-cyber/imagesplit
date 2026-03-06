import os
from PIL import Image
from typing import List, Tuple
import math


def stitch_images_horizontal(image_paths: List[str], output_path: str) -> str:
    images = []
    min_height = float('inf')
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
            min_height = min(min_height, img.height)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images to stitch")
    
    min_height = int(min_height)
    
    resized_images = []
    for img in images:
        if img.height != min_height:
            ratio = min_height / img.height
            new_width = int(img.width * ratio)
            img = img.resize((new_width, min_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
    
    total_width = sum(img.width for img in resized_images)
    stitched_image = Image.new('RGB', (total_width, min_height))
    
    x_offset = 0
    for img in resized_images:
        stitched_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stitched_image.save(output_path, quality=95)
    
    for img in images:
        img.close()
    
    return output_path


def stitch_images_vertical(image_paths: List[str], output_path: str) -> str:
    images = []
    min_width = float('inf')
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
            min_width = min(min_width, img.width)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images to stitch")
    
    min_width = int(min_width)
    
    resized_images = []
    for img in images:
        if img.width != min_width:
            ratio = min_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((min_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
    
    total_height = sum(img.height for img in resized_images)
    stitched_image = Image.new('RGB', (min_width, total_height))
    
    y_offset = 0
    for img in resized_images:
        stitched_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stitched_image.save(output_path, quality=95)
    
    for img in images:
        img.close()
    
    return output_path


def stitch_images_grid(image_paths: List[str], output_path: str, cols: int = 2) -> str:
    n_images = len(image_paths)
    if n_images == 0:
        raise ValueError("No images to stitch")
    
    rows = math.ceil(n_images / cols)
    
    images = []
    min_width = float('inf')
    min_height = float('inf')
    
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            images.append(img)
            min_width = min(min_width, img.width)
            min_height = min(min_height, img.height)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images to stitch")
    
    min_width = int(min_width)
    min_height = int(min_height)
    
    resized_images = []
    for img in images:
        if img.width != min_width or img.height != min_height:
            img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
        resized_images.append(img)
    
    total_width = cols * min_width
    total_height = rows * min_height
    stitched_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    for idx, img in enumerate(resized_images):
        row = idx // cols
        col = idx % cols
        x_offset = col * min_width
        y_offset = row * min_height
        stitched_image.paste(img, (x_offset, y_offset))
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stitched_image.save(output_path, quality=95)
    
    for img in images:
        img.close()
    
    return output_path
