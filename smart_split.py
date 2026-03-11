import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import math


class SmartSplitDetector:
    def __init__(self):
        self.min_panel_size = 100
        self.overlap_margin = 30
    
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
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"[DEBUG] PIL fallback succeeded")
            except Exception as e:
                print(f"[ERROR] Cannot read image: {e}")
                return []
        
        height, width = img.shape[:2]
        print(f"[DEBUG] Image dimensions: {width}x{height}")
        
        # 检测分割线
        h_split_lines, v_split_lines = self._detect_main_split_lines(img)
        
        print(f"[DEBUG] Horizontal split lines: {h_split_lines}")
        print(f"[DEBUG] Vertical split lines: {v_split_lines}")
        
        # 创建面板
        panels = self._create_panels(h_split_lines, v_split_lines, width, height)
        
        # 添加重叠
        panels = self._add_overlap(panels, width, height)
        
        # 过滤太小的面板
        panels = [p for p in panels if (p[2] - p[0]) >= self.min_panel_size and (p[3] - p[1]) >= self.min_panel_size]
        
        if not panels:
            panels = [(0, 0, width, height)]
        
        return panels
    
    def _detect_main_split_lines(self, img: np.ndarray) -> Tuple[List[int], List[int]]:
        """检测主要的分割线"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 方法1: 检测明显的空白分隔线
        h_lines1, v_lines1 = self._detect_whitespace_lines(gray)
        
        # 方法2: 检测边缘分隔
        h_lines2, v_lines2 = self._detect_edge_lines(gray)
        
        # 合并结果，优先使用空白检测
        h_lines = h_lines1 if len(h_lines1) >= len(h_lines2) else h_lines2
        v_lines = v_lines1 if len(v_lines1) >= len(v_lines2) else v_lines2
        
        return h_lines, v_lines
    
    def _detect_whitespace_lines(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """检测空白分隔线"""
        height, width = gray.shape
        
        # 检测白色或接近白色的背景
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # 水平投影 - 每行白色像素数量
        h_projection = np.sum(binary, axis=1)
        h_max = width * 255
        h_threshold = h_max * 0.95  # 95%以上是白色
        
        # 找到连续的空白行
        h_lines = self._find_continuous_gaps(h_projection, h_threshold, height, min_continuous=20)
        
        # 垂直投影
        v_projection = np.sum(binary, axis=0)
        v_max = height * 255
        v_threshold = v_max * 0.95
        
        v_lines = self._find_continuous_gaps(v_projection, v_threshold, width, min_continuous=20)
        
        return h_lines, v_lines
    
    def _detect_edge_lines(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """通过边缘密度检测分割线"""
        height, width = gray.shape
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 水平边缘密度
        h_edges = np.sum(edges, axis=1)
        h_threshold = np.mean(h_edges) * 0.2
        
        # 找低边缘密度的区域
        h_lines = self._find_low_density_regions(h_edges, h_threshold, min_width=30)
        
        # 垂直边缘密度
        v_edges = np.sum(edges, axis=0)
        v_threshold = np.mean(v_edges) * 0.2
        
        v_lines = self._find_low_density_regions(v_edges, v_threshold, min_width=30)
        
        return h_lines, v_lines
    
    def _find_continuous_gaps(self, projection: np.ndarray, threshold: float, 
                               total_length: int, min_continuous: int = 20) -> List[int]:
        """找到连续的空白区域，返回分割线位置"""
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
                    if gap_len >= min_continuous:
                        # 分割线在空白区域的中间
                        gaps.append(gap_start + gap_len // 2)
                    in_gap = False
        
        return gaps
    
    def _find_low_density_regions(self, data: np.ndarray, threshold: float, 
                                    min_width: int = 30) -> List[int]:
        """找到低密度区域"""
        lines = []
        in_low = False
        low_start = 0
        
        for i, val in enumerate(data):
            if val < threshold:
                if not in_low:
                    in_low = True
                    low_start = i
            else:
                if in_low:
                    width = i - low_start
                    if width >= min_width:
                        lines.append(low_start + width // 2)
                    in_low = False
        
        return lines
    
    def _create_panels(self, h_lines: List[int], v_lines: List[int], 
                        width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """根据分割线创建面板"""
        panels = []
        
        if not h_lines and not v_lines:
            # 没有分割线，返回整图
            return [(0, 0, width, height)]
        
        if v_lines and not h_lines:
            # 只有垂直分割
            x_positions = [0] + v_lines + [width]
            for i in range(len(x_positions) - 1):
                panels.append((x_positions[i], 0, x_positions[i + 1], height))
        elif h_lines and not v_lines:
            # 只有水平分割
            y_positions = [0] + h_lines + [height]
            for i in range(len(y_positions) - 1):
                panels.append((0, y_positions[i], width, y_positions[i + 1]))
        elif h_lines and v_lines:
            # 网格分割
            x_positions = [0] + v_lines + [width]
            y_positions = [0] + h_lines + [height]
            for yi in range(len(y_positions) - 1):
                for xi in range(len(x_positions) - 1):
                    panels.append((
                        x_positions[xi],
                        y_positions[yi],
                        x_positions[xi + 1],
                        y_positions[yi + 1]
                    ))
        
        return panels
    
    def _add_overlap(self, panels: List[Tuple[int, int, int, int]], 
                      width: int, height: int) -> List[Tuple[int, int, int, int]]:
        """为相邻面板添加重叠区域"""
        if len(panels) <= 1:
            return panels
        
        # 判断是水平还是垂直排列
        if len(panels) == 1:
            return panels
        
        # 检查是否水平排列（y坐标相近）
        y_starts = [p[1] for p in panels]
        y_same = max(y_starts) - min(y_starts) < self.min_panel_size
        
        # 检查是否垂直排列（x坐标相近）
        x_starts = [p[0] for p in panels]
        x_same = max(x_starts) - min(x_starts) < self.min_panel_size
        
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
            # 网格排列，按位置排序
            panels = sorted(panels, key=lambda p: (p[1], p[0]))
            
            # 找出唯一的x和y边界
            x_boundaries = sorted(set([p[0] for p in panels] + [p[2] for p in panels]))
            y_boundaries = sorted(set([p[1] for p in panels] + [p[3] for p in panels]))
            
            for x1, y1, x2, y2 in panels:
                new_x1 = max(0, x1 - self.overlap_margin)
                new_y1 = max(0, y1 - self.overlap_margin)
                new_x2 = min(width, x2 + self.overlap_margin)
                new_y2 = min(height, y2 + self.overlap_margin)
                overlapped.append((new_x1, new_y1, new_x2, new_y2))
        
        return overlapped
    
    def detect_split_lines(self, image_path: str, min_gap: int = 20) -> Tuple[List[int], List[int], str]:
        img = cv2.imread(image_path)
        if img is None:
            return [], [], "horizontal"
        
        h_lines, v_lines = self._detect_main_split_lines(img)
        
        if len(v_lines) >= len(h_lines):
            mode = "horizontal"
        else:
            mode = "vertical"
        
        return h_lines, v_lines, mode
    
    def detect_panels_by_content(self, image_path: str, padding: int = 3) -> List[Tuple[int, int, int, int]]:
        self.overlap_margin = padding
        return self.detect_all_panels(image_path)
