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
        """检测所有面板区域"""
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
        print(f"[DEBUG] Background intensity: {bg_color}")
        fg_mask = self._create_foreground_mask(gray, bg_color)

        # 主检测
        h_lines, v_lines = self._detect_by_projection_valleys(fg_mask, width, height)

        # 回退：连通域间隙
        if not h_lines and not v_lines:
            h_lines, v_lines = self._detect_by_component_gaps(fg_mask, width, height)

        # 回退：传统空白行
        if not h_lines and not v_lines:
            h_lines, v_lines = self._detect_whitespace_lines(gray, bg_color)

        print(f"[DEBUG] Split lines -> h={h_lines}, v={v_lines}")

        panels = self._create_panels(h_lines, v_lines, width, height)

        # ★ 水平分割后对每个宽子区递归做垂直检测（解决001类问题）
        if h_lines and not v_lines:
            panels = self._recursive_vsplit(fg_mask, panels, width)

        panels = self._add_overlap(panels, width, height)
        panels = [p for p in panels
                  if (p[2]-p[0]) >= self.min_panel_size and (p[3]-p[1]) >= self.min_panel_size]
        panels = [p for p in panels if not self._is_blank_panel(fg_mask, p)]
        if not panels:
            panels = [(0, 0, width, height)]
        return panels

    # ═══════════════════════════════════════════════════════
    # 策略1：投影谷值检测
    # ═══════════════════════════════════════════════════════

    def _detect_by_projection_valleys(
        self, fg_mask: np.ndarray, width: int, height: int
    ) -> Tuple[List[int], List[int]]:
        v_proj = np.sum(fg_mask > 0, axis=0).astype(np.float32)
        h_proj = np.sum(fg_mask > 0, axis=1).astype(np.float32)

        min_dist_v = max(60, width // 8)
        min_dist_h = max(60, height // 8)

        v_lines = self._find_valleys(v_proj, min_distance=min_dist_v)
        h_lines = self._find_valleys(h_proj, min_distance=min_dist_h)

        v_lines, h_lines = self._select_direction(v_proj, h_proj, v_lines, h_lines, width, height)
        return h_lines, v_lines

    def _find_valleys(
        self,
        projection: np.ndarray,
        min_distance: int,
        min_prominence_ratio: float = 0.35,
        abs_ratio_max: float = 0.50,
    ) -> List[int]:
        """
        在投影中找显著谷值。
        过滤条件（同时满足）：
          1. 突出度 >= max * min_prominence_ratio（谷足够深）
          2. 谷值 / max < abs_ratio_max（谷绝对量足够低，非物体内部细颈）
        相邻谷值合并，保留突出度更大的。
        """
        n = len(projection)
        margin = max(5, int(n * 0.04))
        sigma = max(8, n // 35)
        ks = sigma * 6 + 1
        if ks % 2 == 0:
            ks += 1
        smoothed = cv2.GaussianBlur(
            projection.reshape(1, -1), (ks, 1), sigmaX=sigma
        ).flatten()

        max_val = float(smoothed.max())
        if max_val < 1:
            return []

        candidates: List[Tuple[int, float]] = []
        for i in range(1, n - 1):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                if not (margin < i < n - margin):
                    continue
                left_peak = float(smoothed[:i].max())
                right_peak = float(smoothed[i + 1:].max())
                prominence = min(left_peak - smoothed[i], right_peak - smoothed[i])
                abs_ratio = smoothed[i] / max_val
                # 同时检查突出度和绝对量
                if (prominence >= max_val * min_prominence_ratio
                        and abs_ratio < abs_ratio_max):
                    candidates.append((i, prominence))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0])
        merged: List[Tuple[int, float]] = [candidates[0]]
        for pos, prom in candidates[1:]:
            if pos - merged[-1][0] >= min_distance:
                merged.append((pos, prom))
            elif prom > merged[-1][1]:
                merged[-1] = (pos, prom)

        result = [pos for pos, _ in merged]
        print(f"[DEBUG] Valley candidates: {result}")
        return result

    def _select_direction(
        self,
        v_proj: np.ndarray, h_proj: np.ndarray,
        v_lines: List[int], h_lines: List[int],
        width: int, height: int,
    ) -> Tuple[List[int], List[int]]:
        """
        方向选择策略：
          1. 只有一个方向有谷值：直接用该方向
          2. 两个方向都有谷值：
             a. 数量不等 → 选数量多的
             b. 数量相等 → 若两者平均突出度均 >= 40%，允许网格分割
             c. 数量相等 → 其他情况选更深的；平均深度相差 < 5% 时宽图偏V
        """
        if not v_lines and not h_lines:
            return [], []
        if v_lines and not h_lines:
            return v_lines, []
        if h_lines and not v_lines:
            return [], h_lines

        # 两方向都有谷值
        vc, hc = len(v_lines), len(h_lines)
        if vc > hc:
            print(f"[DEBUG] Direction: V wins by count ({vc} vs {hc})")
            return v_lines, []
        if hc > vc:
            print(f"[DEBUG] Direction: H wins by count ({hc} vs {vc})")
            return [], h_lines

        # 数量相等，计算平均突出度
        v_depth = self._avg_depth(v_proj, v_lines)
        h_depth = self._avg_depth(h_proj, h_lines)

        # 两方向均显著 → 允许网格（解决010类问题）
        if v_depth >= 0.40 and h_depth >= 0.40:
            print(f"[DEBUG] Direction: GRID allowed (v_depth={v_depth:.2f}, h_depth={h_depth:.2f})")
            return v_lines, h_lines

        # 选更深的
        diff = abs(v_depth - h_depth)
        if diff < 0.05 and width / height > 1.5:
            # 宽图且深度接近 → 偏向V（物体并排更常见）
            print(f"[DEBUG] Direction: V preferred (wide image, similar depth)")
            return v_lines, []
        if v_depth >= h_depth:
            print(f"[DEBUG] Direction: V wins by depth ({v_depth:.2f} vs {h_depth:.2f})")
            return v_lines, []
        else:
            print(f"[DEBUG] Direction: H wins by depth ({h_depth:.2f} vs {v_depth:.2f})")
            return [], h_lines

    def _avg_depth(self, projection: np.ndarray, positions: List[int]) -> float:
        """计算一组谷值的平均突出度比例"""
        if not positions:
            return 0.0
        max_val = float(projection.max())
        if max_val == 0:
            return 0.0
        depths = []
        for pos in positions:
            lp = float(projection[:pos].max()) if pos > 0 else 0.0
            rp = float(projection[pos+1:].max()) if pos < len(projection)-1 else 0.0
            depths.append(min(lp, rp) - float(projection[pos]))
        return sum(depths) / len(depths) / max_val

    # ═══════════════════════════════════════════════════════
    # ★ 递归竖向分割（解决 001 类：水平分割后各行再竖向分割）
    # ═══════════════════════════════════════════════════════

    def _recursive_vsplit(
        self,
        fg_mask: np.ndarray,
        panels: List[Tuple[int, int, int, int]],
        total_width: int,
    ) -> List[Tuple[int, int, int, int]]:
        """
        对每个宽度远大于高度的子面板，独立运行竖向谷值检测。
        用于处理"先水平分割，各行再各自竖向分割"的布局。
        """
        new_panels = []
        for (x1, y1, x2, y2) in panels:
            pw = x2 - x1
            ph = y2 - y1
            if ph == 0:
                new_panels.append((x1, y1, x2, y2))
                continue
            # 只对宽高比 > 1.2 的子面板尝试竖向分割
            if pw / ph > 1.2:
                sub_fg = fg_mask[y1:y2, x1:x2]
                v_proj = np.sum(sub_fg > 0, axis=0).astype(np.float32)
                min_dist = max(60, pw // 8)
                v_sub = self._find_valleys(v_proj, min_distance=min_dist)
                if v_sub:
                    print(f"[DEBUG] Recursive vsplit for ({x1},{y1})-({x2},{y2}): {v_sub}")
                    x_pos = [x1] + [x1 + vl for vl in v_sub] + [x2]
                    for i in range(len(x_pos) - 1):
                        new_panels.append((x_pos[i], y1, x_pos[i+1], y2))
                    continue
            new_panels.append((x1, y1, x2, y2))
        return new_panels

    # ═══════════════════════════════════════════════════════
    # 策略2：连通域间隙法
    # ═══════════════════════════════════════════════════════

    def _detect_by_component_gaps(
        self, fg_mask: np.ndarray, width: int, height: int
    ) -> Tuple[List[int], List[int]]:
        close_px = max(8, min(width, height) // 60)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)
        )
        closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        min_area = width * height * 0.003
        bboxes = []
        for i in range(1, num_labels):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                x1 = int(stats[i, cv2.CC_STAT_LEFT])
                y1 = int(stats[i, cv2.CC_STAT_TOP])
                w = int(stats[i, cv2.CC_STAT_WIDTH])
                h = int(stats[i, cv2.CC_STAT_HEIGHT])
                bboxes.append((x1, y1, x1 + w, y1 + h))
        if len(bboxes) <= 1:
            return [], []
        v_lines = self._gaps_from_ranges(
            [b[0] for b in bboxes], [b[2] for b in bboxes], total=width
        )
        h_lines = self._gaps_from_ranges(
            [b[1] for b in bboxes], [b[3] for b in bboxes], total=height
        )
        return h_lines, v_lines

    def _gaps_from_ranges(
        self, starts: List[int], ends: List[int], total: int, min_gap: int = 5
    ) -> List[int]:
        margin = max(3, int(total * 0.03))
        ranges = sorted(zip(starts, ends), key=lambda r: r[0])
        merged: List[Tuple[int, int]] = []
        for s, e in ranges:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        gaps = []
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i + 1][0]
            gap_len = gap_end - gap_start
            mid = gap_start + gap_len // 2
            if gap_len >= min_gap and margin < mid < total - margin:
                gaps.append(mid)
        return gaps

    # ═══════════════════════════════════════════════════════
    # 策略3：传统空白行法（兜底）
    # ═══════════════════════════════════════════════════════

    def _detect_whitespace_lines(
        self, gray: np.ndarray, bg_color: int
    ) -> Tuple[List[int], List[int]]:
        height, width = gray.shape
        diff = np.abs(gray.astype(np.int32) - bg_color)
        near_bg = np.where(diff <= 20, np.uint8(255), np.uint8(0))
        h_proj = np.sum(near_bg, axis=1).astype(float)
        v_proj = np.sum(near_bg, axis=0).astype(float)
        h_lines = self._find_continuous_gaps_threshold(h_proj, width * 255 * 0.95, height)
        v_lines = self._find_continuous_gaps_threshold(v_proj, height * 255 * 0.95, width)
        return h_lines, v_lines

    def _find_continuous_gaps_threshold(
        self, projection: np.ndarray, threshold: float, total_length: int,
        min_continuous: int = 15
    ) -> List[int]:
        margin = max(5, int(total_length * 0.05))
        gaps, in_gap, gap_start = [], False, 0
        for i, val in enumerate(projection):
            if val >= threshold:
                if not in_gap:
                    in_gap, gap_start = True, i
            elif in_gap:
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

    # ═══════════════════════════════════════════════════════
    # 背景检测 & 前景遮罩
    # ═══════════════════════════════════════════════════════

    def _detect_background_color(self, gray: np.ndarray) -> int:
        h, w = gray.shape
        border_size = max(10, min(h, w) // 20)
        border_pixels = np.concatenate([
            gray[:border_size, :].flatten(),
            gray[-border_size:, :].flatten(),
            gray[:, :border_size].flatten(),
            gray[:, -border_size:].flatten(),
        ])
        return int(np.median(border_pixels))

    def _create_foreground_mask(
        self, gray: np.ndarray, bg_color: int, tolerance: int = 15
    ) -> np.ndarray:
        """全局差值 + 局部对比度（大核高斯），取最大，对渐变背景鲁棒"""
        h, w = gray.shape
        diff_global = np.abs(gray.astype(np.int32) - bg_color).astype(np.uint8)
        blur_size = max(51, min(h, w) // 8)
        if blur_size % 2 == 0:
            blur_size += 1
        local_bg = cv2.GaussianBlur(gray.astype(np.float32), (blur_size, blur_size), 0)
        diff_local = np.abs(gray.astype(np.float32) - local_bg).clip(0, 255).astype(np.uint8)
        combined = np.maximum(diff_global, diff_local)
        mask = np.where(combined > tolerance, np.uint8(255), np.uint8(0))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # ═══════════════════════════════════════════════════════
    # 空白面板过滤
    # ═══════════════════════════════════════════════════════

    def _is_blank_panel(
        self, fg_mask: np.ndarray, panel: Tuple[int, int, int, int],
        min_content_ratio: float = 0.005,
    ) -> bool:
        x1, y1, x2, y2 = panel
        region = fg_mask[y1:y2, x1:x2]
        if region.size == 0:
            return True
        ratio = float(np.sum(region > 0)) / (region.shape[0] * region.shape[1])
        print(f"[DEBUG] Panel ({x1},{y1})-({x2},{y2}) content ratio: {ratio:.3f}")
        return ratio < min_content_ratio

    # ═══════════════════════════════════════════════════════
    # 面板创建与重叠
    # ═══════════════════════════════════════════════════════

    def _create_panels(
        self, h_lines: List[int], v_lines: List[int], width: int, height: int
    ) -> List[Tuple[int, int, int, int]]:
        if not h_lines and not v_lines:
            return [(0, 0, width, height)]
        if v_lines and not h_lines:
            x_pos = [0] + sorted(v_lines) + [width]
            return [(x_pos[i], 0, x_pos[i+1], height) for i in range(len(x_pos)-1)]
        if h_lines and not v_lines:
            y_pos = [0] + sorted(h_lines) + [height]
            return [(0, y_pos[i], width, y_pos[i+1]) for i in range(len(y_pos)-1)]
        x_pos = [0] + sorted(v_lines) + [width]
        y_pos = [0] + sorted(h_lines) + [height]
        return [
            (x_pos[xi], y_pos[yi], x_pos[xi+1], y_pos[yi+1])
            for yi in range(len(y_pos)-1)
            for xi in range(len(x_pos)-1)
        ]

    def _add_overlap(
        self, panels: List[Tuple[int, int, int, int]], width: int, height: int
    ) -> List[Tuple[int, int, int, int]]:
        if len(panels) <= 1:
            return panels
        y_starts = [p[1] for p in panels]
        x_starts = [p[0] for p in panels]
        y_same = (max(y_starts) - min(y_starts)) < self.min_panel_size
        x_same = (max(x_starts) - min(x_starts)) < self.min_panel_size
        overlapped = []
        if y_same and not x_same:
            panels = sorted(panels, key=lambda p: p[0])
            for i, (x1, y1, x2, y2) in enumerate(panels):
                overlapped.append((
                    max(0, x1-self.overlap_margin) if i > 0 else x1,
                    y1,
                    min(width, x2+self.overlap_margin) if i < len(panels)-1 else x2,
                    y2,
                ))
        elif x_same and not y_same:
            panels = sorted(panels, key=lambda p: p[1])
            for i, (x1, y1, x2, y2) in enumerate(panels):
                overlapped.append((
                    x1,
                    max(0, y1-self.overlap_margin) if i > 0 else y1,
                    x2,
                    min(height, y2+self.overlap_margin) if i < len(panels)-1 else y2,
                ))
        else:
            for x1, y1, x2, y2 in sorted(panels, key=lambda p: (p[1], p[0])):
                overlapped.append((
                    max(0, x1-self.overlap_margin),
                    max(0, y1-self.overlap_margin),
                    min(width, x2+self.overlap_margin),
                    min(height, y2+self.overlap_margin),
                ))
        return overlapped

    # ═══════════════════════════════════════════════════════
    # 对外辅助接口（向后兼容）
    # ═══════════════════════════════════════════════════════

    def detect_split_lines(
        self, image_path: str, min_gap: int = 20
    ) -> Tuple[List[int], List[int], str]:
        img = cv2.imread(image_path)
        if img is None:
            return [], [], "horizontal"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_color = self._detect_background_color(gray)
        fg_mask = self._create_foreground_mask(gray, bg_color)
        height, width = gray.shape
        h_lines, v_lines = self._detect_by_projection_valleys(fg_mask, width, height)
        mode = "horizontal" if len(v_lines) >= len(h_lines) else "vertical"
        return h_lines, v_lines, mode

    def detect_panels_by_content(
        self, image_path: str, padding: int = 3
    ) -> List[Tuple[int, int, int, int]]:
        self.overlap_margin = padding
        return self.detect_all_panels(image_path)
