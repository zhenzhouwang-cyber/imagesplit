import os
import math
from PIL import Image
from typing import List, Tuple


def _make_cut_positions(total: int, n_parts: int) -> List[Tuple[int, int]]:
    """
    计算均匀切分的 (start, end) 像素坐标列表。
    使用浮点均分再取整，避免 total % n_parts != 0 时末块丢像素。
    """
    positions = []
    for i in range(n_parts):
        start = round(i * total / n_parts)
        end = round((i + 1) * total / n_parts)
        positions.append((start, end))
    return positions


def split_image_horizontal(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    """水平方向（沿X轴）均匀拆分为 n_parts 列"""
    img = Image.open(image_path)
    width, height = img.size

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_paths = []
    for i, (left, right) in enumerate(_make_cut_positions(width, n_parts)):
        part = img.crop((left, 0, right, height))
        output_filename = f"{base_name}({i + 1}).png"
        output_path_full = os.path.join(output_dir, output_filename)
        part.save(output_path_full)
        output_paths.append(output_path_full)

    img.close()
    return output_paths


def split_image_vertical(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    """垂直方向（沿Y轴）均匀拆分为 n_parts 行"""
    img = Image.open(image_path)
    width, height = img.size

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_paths = []
    for i, (top, bottom) in enumerate(_make_cut_positions(height, n_parts)):
        part = img.crop((0, top, width, bottom))
        output_filename = f"{base_name}({i + 1}).png"
        output_path_full = os.path.join(output_dir, output_filename)
        part.save(output_path_full)
        output_paths.append(output_path_full)

    img.close()
    return output_paths


def split_image_grid(image_path: str, rows: int, cols: int, output_dir: str, base_name: str) -> List[str]:
    """网格拆分为 rows×cols 块，每块尺寸精确对齐（无像素遗漏）"""
    img = Image.open(image_path)
    width, height = img.size

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    col_cuts = _make_cut_positions(width, cols)
    row_cuts = _make_cut_positions(height, rows)

    output_paths = []
    idx = 1
    for top, bottom in row_cuts:
        for left, right in col_cuts:
            part = img.crop((left, top, right, bottom))
            output_filename = f"{base_name}({idx}).png"
            output_path_full = os.path.join(output_dir, output_filename)
            part.save(output_path_full)
            output_paths.append(output_path_full)
            idx += 1

    img.close()
    return output_paths


def auto_split_image(image_path: str, n_parts: int, output_dir: str, base_name: str) -> List[str]:
    """根据宽高比自动选择水平/垂直/网格拆分"""
    with Image.open(image_path) as img:
        width, height = img.size

    aspect_ratio = width / height

    if aspect_ratio > 1.5:
        return split_image_horizontal(image_path, n_parts, output_dir, base_name)
    elif aspect_ratio < 0.67:
        return split_image_vertical(image_path, n_parts, output_dir, base_name)
    else:
        cols = math.ceil(math.sqrt(n_parts))
        rows = math.ceil(n_parts / cols)
        return split_image_grid(image_path, rows, cols, output_dir, base_name)
