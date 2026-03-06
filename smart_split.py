import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import math


class SmartSplitDetector:
    def __init__(self):
        self.min_panel_size = 50
    
    def smart_split(self, image_path: str, output_dir: str, base_name: str) -> List[str]:
        print(f"[DEBUG] smart_split called: image={image_path}, output_dir={output_dir}, base_name={base_name}")
        
        panels = self.detect_all_panels(image_path)
        print(f"[DEBUG] Detected {len(panels)} panels")
        
        if not panels:
            print(f"[DEBUG] No panels detected, returning empty list")
            return []
        
        try:
            image = Image.open(image_path)
            print(f"[DEBUG] Image opened successfully: {image.size}")
        except Exception as e:
            print(f"[DEBUG] Failed to open image: {e}")
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
                print(f"[DEBUG] Saved: {output_path_full}")
            except Exception as e:
                print(f"[DEBUG] Failed to save panel {idx}: {e}")
        
        image.close()
        print(f"[DEBUG] smart_split complete, {len(output_paths)} files saved")
        return output_paths
    
    def detect_all_panels(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        print(f"[DEBUG] detect_all_panels: {image_path}")
        print(f"[DEBUG] File exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"[ERROR] Image file does not exist: {image_path}")
            return []
        
        img = cv2.imread(image_path)
        print(f"[DEBUG] cv2.imread result: {img is not None}")
        
        if img is None:
            print(f"[ERROR] Cannot read image with cv2: {image_path}")
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"[DEBUG] PIL fallback succeeded, shape: {img.shape}")
            except Exception as e:
                print(f"[ERROR] PIL fallback also failed: {e}")
                return []
        
        height, width = img.shape[:2]
        print(f"[DEBUG] Image dimensions: {width}x{height}")
        
        panels = []
        
        panels = self._detect_panels_by_white_space(img)
        
        if len(panels) <= 1:
            panels = self._detect_panels_by_contours(img)
        
        if len(panels) <= 1:
            panels = self._detect_panels_by_edges(img)
        
        if not panels:
            panels = [(0, 0, width, height)]
        
        panels = self._merge_overlapping_panels(panels)
        
        panels = [p for p in panels if (p[2] - p[0]) >= self.min_panel_size and (p[3] - p[1]) >= self.min_panel_size]
        
        return panels
    
    def _detect_panels_by_white_space(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """通过检测空白区域来分割面板"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        h_projection = np.sum(binary, axis=1)
        v_projection = np.sum(binary, axis=0)
        
        h_threshold = width * 255 * 0.9
        v_threshold = height * 255 * 0.9
        
        h_gaps = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(h_projection):
            if val >= h_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    if i - gap_start >= 10:
                        h_gaps.append((gap_start + i) // 2)
                    in_gap = False
        
        v_gaps = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(v_projection):
            if val >= v_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    if i - gap_start >= 10:
                        v_gaps.append((gap_start + i) // 2)
                    in_gap = False
        
        panels = []
        
        if v_gaps and not h_gaps:
            x_positions = [0] + v_gaps + [width]
            for i in range(len(x_positions) - 1):
                if x_positions[i + 1] - x_positions[i] >= self.min_panel_size:
                    panels.append((x_positions[i], 0, x_positions[i + 1], height))
        elif h_gaps and not v_gaps:
            y_positions = [0] + h_gaps + [height]
            for i in range(len(y_positions) - 1):
                if y_positions[i + 1] - y_positions[i] >= self.min_panel_size:
                    panels.append((0, y_positions[i], width, y_positions[i + 1]))
        elif h_gaps and v_gaps:
            x_positions = [0] + v_gaps + [width]
            y_positions = [0] + h_gaps + [height]
            for yi in range(len(y_positions) - 1):
                for xi in range(len(x_positions) - 1):
                    x1, y1 = x_positions[xi], y_positions[yi]
                    x2, y2 = x_positions[xi + 1], y_positions[yi + 1]
                    if x2 - x1 >= self.min_panel_size and y2 - y1 >= self.min_panel_size:
                        panels.append((x1, y1, x2, y2))
        
        return panels
    
    def _detect_panels_by_contours(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """通过轮廓检测面板"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        panels = []
        min_area = height * width * 0.005
        padding = 5
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area:
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                panels.append((x1, y1, x2, y2))
        
        return panels
    
    def _detect_panels_by_edges(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """通过边缘和形态学操作检测面板"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        abs_sobel_x = np.abs(sobel_x).astype(np.uint8)
        abs_sobel_y = np.abs(sobel_y).astype(np.uint8)
        
        h_edges = np.sum(abs_sobel_y, axis=1)
        v_edges = np.sum(abs_sobel_x, axis=0)
        
        h_threshold = np.mean(h_edges) + np.std(h_edges) * 0.5
        v_threshold = np.mean(v_edges) + np.std(v_edges) * 0.5
        
        h_lines = []
        for i in range(1, len(h_edges) - 1):
            if h_edges[i] > h_threshold and h_edges[i] > h_edges[i-1] and h_edges[i] > h_edges[i+1]:
                h_lines.append(i)
        
        v_lines = []
        for i in range(1, len(v_edges) - 1):
            if v_edges[i] > v_threshold and v_edges[i] > v_edges[i-1] and v_edges[i] > v_edges[i+1]:
                v_lines.append(i)
        
        h_lines = self._merge_close_lines(h_lines, 30)
        v_lines = self._merge_close_lines(v_lines, 30)
        
        panels = []
        
        if v_lines and not h_lines:
            x_positions = [0] + v_lines + [width]
            for i in range(len(x_positions) - 1):
                if x_positions[i + 1] - x_positions[i] >= self.min_panel_size:
                    panels.append((x_positions[i], 0, x_positions[i + 1], height))
        elif h_lines and not v_lines:
            y_positions = [0] + h_lines + [height]
            for i in range(len(y_positions) - 1):
                if y_positions[i + 1] - y_positions[i] >= self.min_panel_size:
                    panels.append((0, y_positions[i], width, y_positions[i + 1]))
        elif h_lines and v_lines:
            x_positions = [0] + v_lines + [width]
            y_positions = [0] + h_lines + [height]
            for yi in range(len(y_positions) - 1):
                for xi in range(len(x_positions) - 1):
                    x1, y1 = x_positions[xi], y_positions[yi]
                    x2, y2 = x_positions[xi + 1], y_positions[yi + 1]
                    if x2 - x1 >= self.min_panel_size and y2 - y1 >= self.min_panel_size:
                        panels.append((x1, y1, x2, y2))
        
        return panels
    
    def _merge_close_lines(self, lines: List[int], min_gap: int) -> List[int]:
        if not lines:
            return []
        
        lines = sorted(lines)
        merged = [lines[0]]
        
        for line in lines[1:]:
            if line - merged[-1] >= min_gap:
                merged.append(line)
            else:
                merged[-1] = (merged[-1] + line) // 2
        
        return merged
    
    def _merge_overlapping_panels(self, panels: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        if len(panels) <= 1:
            return panels
        
        panels = sorted(panels, key=lambda p: (p[0], p[1]))
        
        merged = []
        
        for panel in panels:
            if not merged:
                merged.append(list(panel))
                continue
            
            last = merged[-1]
            
            overlap_x = min(last[2], panel[2]) - max(last[0], panel[0])
            overlap_y = min(last[3], panel[3]) - max(last[1], panel[1])
            
            iou = 0
            if overlap_x > 0 and overlap_y > 0:
                intersection = overlap_x * overlap_y
                area1 = (last[2] - last[0]) * (last[3] - last[1])
                area2 = (panel[2] - panel[0]) * (panel[3] - panel[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
            
            if iou > 0.5:
                last[0] = min(last[0], panel[0])
                last[1] = min(last[1], panel[1])
                last[2] = max(last[2], panel[2])
                last[3] = max(last[3], panel[3])
            else:
                merged.append(list(panel))
        
        return [tuple(p) for p in merged]
    
    def detect_split_lines(self, image_path: str, min_gap: int = 20) -> Tuple[List[int], List[int], str]:
        img = cv2.imread(image_path)
        if img is None:
            return [], [], "horizontal"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        h_projection = np.sum(binary, axis=1)
        v_projection = np.sum(binary, axis=0)
        
        h_threshold = width * 255 * 0.9
        v_threshold = height * 255 * 0.9
        
        h_lines = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(h_projection):
            if val >= h_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    if i - gap_start >= min_gap:
                        h_lines.append((gap_start + i) // 2)
                    in_gap = False
        
        v_lines = []
        in_gap = False
        gap_start = 0
        for i, val in enumerate(v_projection):
            if val >= v_threshold:
                if not in_gap:
                    in_gap = True
                    gap_start = i
            else:
                if in_gap:
                    if i - gap_start >= min_gap:
                        v_lines.append((gap_start + i) // 2)
                    in_gap = False
        
        h_lines = self._merge_close_lines(h_lines, min_gap)
        v_lines = self._merge_close_lines(v_lines, min_gap)
        
        if len(v_lines) >= len(h_lines):
            mode = "horizontal"
        else:
            mode = "vertical"
        
        return h_lines, v_lines, mode
    
    def detect_panels_by_content(self, image_path: str, padding: int = 3) -> List[Tuple[int, int, int, int]]:
        return self.detect_all_panels(image_path)
