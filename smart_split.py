import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import math


class SmartSplitDetector:
    def __init__(self):
        self.min_panel_size = 100
        self.overlap_margin = 30

    # ─────────────────────────────────────────────
    # 公共入口
    # ─────────────────────────────────────────────

    def smart_split(self, image_path: str, output_dir: str, base_name: str) -> List[str]:
        print(f"[DEBUG] smart_split: {image_path}")

        panels = self.detect_all_panels(image_path)
        print(f"[DEBUG] Detected {len(panels)} panels")

        if not panels:
            return []

        try:
            image = Image.open(image_path)
            print(f"[DEBUG] Image size: {image.size}")
        except Exception as e:
            print(f"[ERROR] Cannot open image: {e}")
            return []

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        panels.sort(key=lambda p: (p[1], p[0]))

        output_paths = []
        for idx, (x1, y1, x2, y2) in enumerate(panels, 1):
            try:
                cropped = image.crop((x1, y1, x2, y2))
                output_filename = f"{base_name}({idx}).png"
                output_path_full = os.path.join(output_dir, output_filename)
                cropped.save(output_path_full)
                output_paths.append(output_path_full)
                print(f"[DEBUG] Saved panel {idx}: ({x1},{y1})-({x2},{y2}), size={x2-x1}x{y2-y1}")
            except Exception as e:
                print(f"[ERROR] Failed to save panel {idx}: {e}")

        image.close()
        return output_paths

    def detect_all_panels(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """检测所有面板区域（内容感知，确保物体完整拆分）"""
        print(f"[DEBUG] detect_all_panels: {image_path}")

        if not os.path.exists(image_path):
            print(f"[ERROR] File not found: {image_path}")
            return []

        img = cv2.imread(image_path)
        if img is None:
            try:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"[DEBUG] PIL fallback succeeded")
            except Exception as e:
                print(f"[ERROR] Cannot read image: {e}")
                return []

        height, width = img.shape[:2]
        print(f"[DEBUG] Image dimensions: {width}x{height}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_color = self._detect_background_color(gray)
        fg_mask = self._create_foreground_mask(gray, bg_color)

        # 基于前景内容找分割线
        h_lines, v_lines = self._detect_split_lines_by_content(img, gray, bg_color, fg_mask)

        print(f"[DEBUG] Horizontal split lines: {h_lines}")
        print(f"[DEBUG] Vertical split lines: {v_lines}")

        panels = self._create_panels(h_lines, v_lines, width, height)
        panels = self._add_overlap(panels, width, height)

        # 过滤尺寸太小的面板
        panels = [
            p for p in panels
            if (p[2] - p[0]) >= self.min_panel_size and (p[3] - p[1]) >= self.min_panel_size
        ]

        # ★ 过滤空白/无内容面板（解决"多余白图"问题）
        panels = [
            p for p in panels
            if not self._is_blank_panel(fg_mask, p)
        ]

        if not panels:
            panels = [(0, 0, width, height)]

        return panels

    # ─────────────────────────────────────────────
    # 核心：基于前景内容的分割线检测
    # ─────────────────────────────────────────────

    def _detect_split_lines_by_content(
        self,
        img: np.ndarray,
        gray: np.ndarray,
        bg_color: int,
        fg_mask: np.ndarray,
    ) -> Tuple[List[int], List[int]]:
        """
        通过检测前景内容的空白间隔确定分割线：
          1. 形态学膨胀（动态核）合并物体内部细小间隙
          2. 在膨胀后的投影中找完全空白的连续区段
          3. 验证候选线在原始前景遮罩中不穿过任何物体
          4. 失败时回退传统空白行检测
        """
        height, width = gray.shape

        # 动态膨胀核：图像越大核越大，但上限 20px，防止把相邻物体合并
        dil_h = max(5, min(20, height // 80))
        dil_v = max(5, min(20, width // 80))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (1, dil_h * 2 + 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (dil_v * 2 + 1, 1))

        # 纵向膨胀 → 用于检测水平分割线
        dilated_h = cv2.dilate(fg_mask, kernel_h)
        # 横向膨胀 → 用于检测垂直分割线
        dilated_v = cv2.dilate(fg_mask, kernel_v)

        # 每行/每列前景像素数（膨胀后）
        h_proj = np.sum(dilated_h > 0, axis=1).astype(float)
        v_proj = np.sum(dilated_v > 0, axis=0).astype(float)

        # ★ 修复：min_gap 基于图像尺寸动态计算
        h_min_gap = max(8, height // 100)
        v_min_gap = max(8, width // 100)

        h_gaps = self._find_empty_gaps(h_proj, min_gap=h_min_gap)
        v_gaps = self._find_empty_gaps(v_proj, min_gap=v_min_gap)

        # 验证分割线不穿过物体
        h_lines = self._validate_split_lines(h_gaps, fg_mask, axis='h')
        v_lines = self._validate_split_lines(v_gaps, fg_mask, axis='v')

        # 回退：内容检测无结果时用传统空白行检测
        if not h_lines and not v_lines:
            print("[DEBUG] Content-based detection found no lines, falling back to whitespace detection")
            h_lines, v_lines = self._detect_whitespace_lines(gray, bg_color)

        return h_lines, v_lines

    # ─────────────────────────────────────────────
    # 背景检测 & 前景遮罩
    # ─────────────────────────────────────────────

    def _detect_background_color(self, gray: np.ndarray) -> int:
        """取图像四边框像素中位数作为背景亮度，自动适配白/黑/彩色背景"""
        h, w = gray.shape
        border_size = max(10, min(h, w) // 20)
        border_pixels = np.concatenate([
            gray[:border_size, :].flatten(),
            gray[-border_size:, :].flatten(),
            gray[:, :border_size].flatten(),
            gray[:, -border_size:].flatten(),
        ])
        return int(np.median(border_pixels))

    def _create_foreground_mask(self, gray: np.ndarray, bg_color: int, tolerance: int = 15) -> np.ndarray:
        """像素与背景色之差 > tolerance 的为前景（支持浅色/深色背景）"""
        diff = np.abs(gray.astype(np.int32) - bg_color)
        return np.where(diff > tolerance, np.uint8(255), np.uint8(0))

    # ─────────────────────────────────────────────
    # 间隔检测
    # ─────────────────────────────────────────────

    def _find_empty_gaps(self, projection: np.ndarray, min_gap: int = 8) -> List[int]:
        """
        在投影数组中找连续空白区段，返回各段中点位置。
        - 允许微量噪点（≤ 0.5% 宽度的前景像素视为空白）
        - 排除头尾 5% 的边缘，避免把图像边框误判为分割线
        ★ 修复：边距基于投影自身长度计算，不依赖外部 total 参数
        """
        n = len(projection)
        margin = max(5, int(n * 0.05))
        # 噪点容忍：该行/列最多允许 0.5% 的像素为前景仍视为空白
        noise_tol = max(1, int(n * 0.005))

        gaps = []
        in_gap = False
        gap_start = 0

        for i, val in enumerate(projection):
            if val <= noise_tol:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gap_len = i - gap_start
                    mid = gap_start + gap_len // 2
                    if gap_len >= min_gap and margin < mid < n - margin:
                        gaps.append(mid)
                    in_gap = False

        # 末尾收尾
        if in_gap:
            gap_len = n - gap_start
            mid = gap_start + gap_len // 2
            if gap_len >= min_gap and margin < mid < n - margin:
                gaps.append(mid)

        return gaps

    def _validate_split_lines(
        self,
        candidates: List[int],
        fg_mask: np.ndarray,
        axis: str,
    ) -> List[int]:
        """
        验证候选分割线在原始前景遮罩中没有前景像素（不切断物体）。
        若命中物体，则在 ±8px 范围内寻找最近的空白行/列。
        """
        valid = []
        for pos in candidates:
            if axis == 'h':
                line = fg_mask[pos, :]
                if np.sum(line > 0) == 0:
                    valid.append(pos)
                else:
                    found = self._find_nearest_empty_line(fg_mask, pos, search=8, axis='h')
                    if found is not None:
                        valid.append(found)
            else:
                line = fg_mask[:, pos]
                if np.sum(line > 0) == 0:
                    valid.append(pos)
                else:
                    found = self._find_nearest_empty_line(fg_mask, pos, search=8, axis='v')
                    if found is not None:
                        valid.append(found)
        return valid

    def _find_nearest_empty_line(
        self,
        fg_mask: np.ndarray,
        pos: int,
        search: int,
        axis: str,
    ) -> Optional[int]:
        """在 pos ± search 范围内找前景像素为 0 的最近行/列"""
        h, w = fg_mask.shape
        limit = h if axis == 'h' else w
        for delta in range(search + 1):
            for sign in ([0] if delta == 0 else [1, -1]):
                p = pos + sign * delta
                if p < 0 or p >= limit:
                    continue
                line = fg_mask[p, :] if axis == 'h' else fg_mask[:, p]
                if np.sum(line > 0) == 0:
                    return p
        return None

    # ─────────────────────────────────────────────
    # ★ 新增：空白面板过滤
    # ─────────────────────────────────────────────

    def _is_blank_panel(
        self,
        fg_mask: np.ndarray,
        panel: Tuple[int, int, int, int],
        min_content_ratio: float = 0.005,
    ) -> bool:
        """
        检测面板区域是否几乎没有内容（全白/全背景色）。
        min_content_ratio: 前景像素占比低于此值视为空白面板，默认 0.5%。
        """
        x1, y1, x2, y2 = panel
        region = fg_mask[y1:y2, x1:x2]
        if region.size == 0:
            return True
        fg_count = int(np.sum(region > 0))
        total = region.shape[0] * region.shape[1]
        ratio = fg_count / total
        print(f"[DEBUG] Panel ({x1},{y1})-({x2},{y2}) content ratio: {ratio:.3f}")
        return ratio < min_content_ratio

    # ─────────────────────────────────────────────
    # 回退方法：传统空白行检测
    # ─────────────────────────────────────────────

    def _detect_whitespace_lines(
        self, gray: np.ndarray, bg_color: int
    ) -> Tuple[List[int], List[int]]:
        """回退方案：检测接近背景色的连续空白行/列（支持任意背景色）"""
        height, width = gray.shape
        tolerance = 20
        diff = np.abs(gray.astype(np.int32) - bg_color)
        near_bg = np.where(diff <= tolerance, np.uint8(255), np.uint8(0))

        h_proj = np.sum(near_bg, axis=1).astype(float)
        h_threshold = width * 255 * 0.95
        h_lines = self._find_continuous_gaps_threshold(h_proj, h_threshold, height, min_continuous=15)

        v_proj = np.sum(near_bg, axis=0).astype(float)
        v_threshold = height * 255 * 0.95
        v_lines = self._find_continuous_gaps_threshold(v_proj, v_threshold, width, min_continuous=15)

        return h_lines, v_lines

    def _find_continuous_gaps_threshold(
        self,
        projection: np.ndarray,
        threshold: float,
        total_length: int,
        min_continuous: int = 15,
    ) -> List[int]:
        """找连续满足阈值的区域，返回中点（用于回退方案）"""
        margin = max(5, int(total_length * 0.05))
        gaps = []
        in_gap = False
        gap_start = 0

        for i, val in enumerate(projection):
            if val >= threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    gap_len = i - gap_start
                    mid = gap_start + gap_len // 2
                    if gap_len >= min_continuous and margin < mid < total_length - margin:
                        gaps.append(mid)
                    in_gap = False

        if in_gap:
            gap_len = total_length - gap_start
            mid = gap_start + gap_len // 2
            if gap_len >= min_continuous and margin < mid < total_length - margin:
                gaps.append(mid)

        return gaps

    # ─────────────────────────────────────────────
    # 面板创建与重叠
    # ─────────────────────────────────────────────

    def _create_panels(
        self,
        h_lines: List[int],
        v_lines: List[int],
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """根据分割线创建面板边界框"""
        if not h_lines and not v_lines:
            return [(0, 0, width, height)]

        if v_lines and not h_lines:
            x_positions = [0] + sorted(v_lines) + [width]
            return [(x_positions[i], 0, x_positions[i + 1], height)
                    for i in range(len(x_positions) - 1)]

        if h_lines and not v_lines:
            y_positions = [0] + sorted(h_lines) + [height]
            return [(0, y_positions[i], width, y_positions[i + 1])
                    for i in range(len(y_positions) - 1)]

        # 网格分割
        x_positions = [0] + sorted(v_lines) + [width]
        y_positions = [0] + sorted(h_lines) + [height]
        panels = []
        for yi in range(len(y_positions) - 1):
            for xi in range(len(x_positions) - 1):
                panels.append((
                    x_positions[xi],
                    y_positions[yi],
                    x_positions[xi + 1],
                    y_positions[yi + 1],
                ))
        return panels

    def _add_overlap(
        self,
        panels: List[Tuple[int, int, int, int]],
        width: int,
        height: int,
    ) -> List[Tuple[int, int, int, int]]:
        """为相邻面板添加重叠边距，防止边缘内容被裁切"""
        if len(panels) <= 1:
            return panels

        y_starts = [p[1] for p in panels]
        x_starts = [p[0] for p in panels]
        y_same = (max(y_starts) - min(y_starts)) < self.min_panel_size
        x_same = (max(x_starts) - min(x_starts)) < self.min_panel_size

        overlapped = []

        if y_same and not x_same:
            # 水平排列
            panels = sorted(panels, key=lambda p: p[0])
            for i, (x1, y1, x2, y2) in enumerate(panels):
                new_x1 = max(0, x1 - self.overlap_margin) if i > 0 else x1
                new_x2 = min(width, x2 + self.overlap_margin) if i < len(panels) - 1 else x2
                overlapped.append((new_x1, y1, new_x2, y2))

        elif x_same and not y_same:
            # 垂直排列
            panels = sorted(panels, key=lambda p: p[1])
            for i, (x1, y1, x2, y2) in enumerate(panels):
                new_y1 = max(0, y1 - self.overlap_margin) if i > 0 else y1
                new_y2 = min(height, y2 + self.overlap_margin) if i < len(panels) - 1 else y2
                overlapped.append((x1, new_y1, x2, new_y2))

        else:
            # 网格排列：各方向均加重叠
            panels = sorted(panels, key=lambda p: (p[1], p[0]))
            for x1, y1, x2, y2 in panels:
                overlapped.append((
                    max(0, x1 - self.overlap_margin),
                    max(0, y1 - self.overlap_margin),
                    min(width, x2 + self.overlap_margin),
                    min(height, y2 + self.overlap_margin),
                ))

        return overlapped

    # ─────────────────────────────────────────────
    # 对外辅助接口（保持向后兼容）
    # ─────────────────────────────────────────────

    def detect_split_lines(
        self, image_path: str, min_gap: int = 20
    ) -> Tuple[List[int], List[int], str]:
        img = cv2.imread(image_path)
        if img is None:
            return [], [], "horizontal"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_color = self._detect_background_color(gray)
        fg_mask = self._create_foreground_mask(gray, bg_color)
        h_lines, v_lines = self._detect_split_lines_by_content(img, gray, bg_color, fg_mask)

        mode = "horizontal" if len(v_lines) >= len(h_lines) else "vertical"
        return h_lines, v_lines, mode

    def detect_panels_by_content(
        self, image_path: str, padding: int = 3
    ) -> List[Tuple[int, int, int, int]]:
        self.overlap_margin = padding
        return self.detect_all_panels(image_path)
