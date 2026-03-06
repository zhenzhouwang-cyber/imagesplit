import os
from PIL import Image
from typing import List, Tuple


def split_image_horizontal(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    img = Image.open(image_path)
    width, height = img.size
    part_width = width // n_parts
    
    output_paths = []
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(n_parts):
        left = i * part_width
        right = (i + 1) * part_width if i < n_parts - 1 else width
        
        part = img.crop((left, 0, right, height))
        
        output_filename = f"{base_name}({i+1}).png"
        output_path_full = os.path.join(output_dir, output_filename)
        part.save(output_path_full)
        output_paths.append(output_path_full)
    
    img.close()
    return output_paths


def split_image_vertical(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    img = Image.open(image_path)
    width, height = img.size
    part_height = height // n_parts
    
    output_paths = []
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(n_parts):
        top = i * part_height
        bottom = (i + 1) * part_height if i < n_parts - 1 else height
        
        part = img.crop((0, top, width, bottom))
        
        output_filename = f"{base_name}({i+1}).png"
        output_path_full = os.path.join(output_dir, output_filename)
        part.save(output_path_full)
        output_paths.append(output_path_full)
    
    img.close()
    return output_paths


def split_image_grid(image_path: str, rows: int, cols: int, output_dir: str, base_name: str) -> List[str]:
    img = Image.open(image_path)
    width, height = img.size
    part_width = width // cols
    part_height = height // rows
    
    output_paths = []
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    idx = 1
    for row in range(rows):
        for col in range(cols):
            left = col * part_width
            top = row * part_height
            right = (col + 1) * part_width if col < cols - 1 else width
            bottom = (row + 1) * part_height if row < rows - 1 else height
            
            part = img.crop((left, top, right, bottom))
            
            output_filename = f"{base_name}({idx}).png"
            output_path_full = os.path.join(output_dir, output_filename)
            part.save(output_path_full)
            output_paths.append(output_path_full)
            idx += 1
    
    img.close()
    return output_paths


def auto_split_image(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    img = Image.open(image_path)
    width, height = img.size
    img.close()
    
    aspect_ratio = width / height
    
    if aspect_ratio > 1.5:
        return split_image_horizontal(image_path, n_parts, output_dir, base_name)
    elif aspect_ratio < 0.67:
        return split_image_vertical(image_path, n_parts, output_dir, base_name)
    else:
        import math
        cols = math.ceil(math.sqrt(n_parts))
        rows = math.ceil(n_parts / cols)
        return split_image_grid(image_path, rows, cols, output_dir, base_name)
