import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


class TextRemover:
    def __init__(self):
        self.min_text_area = 50
        self.detection_method = "mser"
    
    def remove_text(self, image_path: str, output_path: str, 
                    dilate_size: int = 5, 
                    inpaint_radius: int = 5,
                    detection_method: str = "auto") -> bool:
        """
        检测并去除图片中的文字
        
        Args:
            image_path: 输入图片路径
            output_path: 输出图片路径
            dilate_size: 膨胀核大小，用于扩展文字区域
            inpaint_radius: 修复半径
            detection_method: 检测方法 "mser", "edge", "auto"
        
        Returns:
            是否成功处理
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Cannot read image: {image_path}")
            return False
        
        height, width = img.shape[:2]
        print(f"[INFO] Image size: {width}x{height}")
        
        if detection_method == "auto":
            text_mask = self._detect_text_auto(img)
        elif detection_method == "mser":
            text_mask = self._detect_text_mser(img)
        elif detection_method == "edge":
            text_mask = self._detect_text_edge(img)
        else:
            text_mask = self._detect_text_auto(img)
        
        if text_mask is None or np.sum(text_mask) == 0:
            print("[INFO] No text detected in image")
            cv2.imwrite(output_path, img)
            return True
        
        text_pixels = np.sum(text_mask > 0)
        total_pixels = height * width
        text_ratio = text_pixels / total_pixels
        print(f"[INFO] Text area ratio: {text_ratio*100:.2f}%")
        
        if text_ratio > 0.5:
            print("[WARN] Text area too large, might be false positive")
            cv2.imwrite(output_path, img)
            return True
        
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        
        result = cv2.inpaint(img, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, result)
        print(f"[INFO] Text removed, saved to: {output_path}")
        
        return True
    
    def _detect_text_auto(self, img: np.ndarray) -> np.ndarray:
        """自动选择最佳检测方法"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mser_mask = self._detect_text_mser(img)
        edge_mask = self._detect_text_edge(img)
        
        mser_score = np.sum(mser_mask > 0) if mser_mask is not None else 0
        edge_score = np.sum(edge_mask > 0) if edge_mask is not None else 0
        
        if mser_score > edge_score:
            print("[INFO] Using MSER detection")
            return mser_mask
        else:
            print("[INFO] Using Edge detection")
            return edge_mask
    
    def _detect_text_mser(self, img: np.ndarray) -> np.ndarray:
        """使用MSER检测文字区域"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        mser = cv2.MSER_create()
        mser.setMinArea(30)
        mser.setMaxArea(int(height * width * 0.1))
        
        regions, _ = mser.detectRegions(gray)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if (10 < area < height * width * 0.05 and 
                0.1 < aspect_ratio < 15 and
                w > 5 and h > 5):
                text_regions.append((x, y, w, h))
        
        text_regions = self._merge_regions(text_regions)
        
        for x, y, w, h in text_regions:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        return mask
    
    def _detect_text_edge(self, img: np.ndarray) -> np.ndarray:
        """使用边缘检测和形态学操作检测文字"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        edges = cv2.Canny(enhanced, 50, 150)
        
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        
        dilated_h = cv2.dilate(edges, kernel_h, iterations=1)
        dilated_v = cv2.dilate(edges, kernel_v, iterations=1)
        
        combined = cv2.bitwise_or(dilated_h, dilated_v)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if (area > 50 and 
                0.1 < aspect_ratio < 20 and
                w > 8 and h > 5):
                text_regions.append((x, y, w, h))
        
        text_regions = self._merge_regions(text_regions)
        
        for x, y, w, h in text_regions:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        return mask
    
    def _merge_regions(self, regions: List[Tuple[int, int, int, int]], 
                       threshold: int = 5) -> List[Tuple[int, int, int, int]]:
        """合并重叠或相邻的区域"""
        if not regions:
            return []
        
        merged = []
        used = [False] * len(regions)
        
        for i, r1 in enumerate(regions):
            if used[i]:
                continue
            
            x1, y1, w1, h1 = r1
            current = [x1, y1, x1 + w1, y1 + h1]
            used[i] = True
            
            changed = True
            while changed:
                changed = False
                for j, r2 in enumerate(regions):
                    if used[j]:
                        continue
                    
                    x2, y2, w2, h2 = r2
                    x2_end, y2_end = x2 + w2, y2 + h2
                    
                    if self._regions_overlap(
                        (current[0] - threshold, current[1] - threshold, 
                         current[2] + threshold, current[3] + threshold),
                        (x2, y2, x2_end, y2_end)
                    ):
                        current[0] = min(current[0], x2)
                        current[1] = min(current[1], y2)
                        current[2] = max(current[2], x2_end)
                        current[3] = max(current[3], y2_end)
                        used[j] = True
                        changed = True
            
            merged.append((current[0], current[1], current[2] - current[0], current[3] - current[1]))
        
        return merged
    
    def _regions_overlap(self, r1: Tuple[int, int, int, int], 
                         r2: Tuple[int, int, int, int]) -> bool:
        """检查两个区域是否重叠"""
        return not (r1[2] < r2[0] or r1[0] > r2[2] or 
                    r1[3] < r2[1] or r1[1] > r2[3])
    
    def get_text_mask(self, image_path: str, detection_method: str = "auto") -> Optional[np.ndarray]:
        """获取文字区域的mask，用于预览"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        if detection_method == "auto":
            return self._detect_text_auto(img)
        elif detection_method == "mser":
            return self._detect_text_mser(img)
        elif detection_method == "edge":
            return self._detect_text_edge(img)
        else:
            return self._detect_text_auto(img)
    
    def preview_text_detection(self, image_path: str, output_path: str, 
                                detection_method: str = "auto") -> bool:
        """生成文字检测预览图（文字区域标红）"""
        img = cv2.imread(image_path)
        if img is None:
            return False
        
        mask = self.get_text_mask(image_path, detection_method)
        if mask is None:
            return False
        
        preview = img.copy()
        preview[mask > 0] = [0, 0, 255]
        
        blended = cv2.addWeighted(img, 0.7, preview, 0.3, 0)
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(output_path, blended)
        return True
