"""
object_remover.py
AI杂物去除模块
使用LaMa等模型进行智能物体移除
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List


class ObjectRemover:
    """
    AI杂物去除器
    支持多种修复方法
    """
    
    def __init__(self, method: str = "lama"):
        """
        初始化
        
        method: 修复方法
            - "lama": LaMa模型（推荐，大范围修复）
            - "cv2": OpenCV inpainting（快速，小范围）
            - "auto": 自动选择
        """
        self.method = method
        self._lama_model = None
    
    def remove_object(
        self,
        image_path: str,
        mask_path: str,
        output_path: str,
        method: str = None,
        inpaint_radius: int = 5,
    ) -> bool:
        """
        根据mask移除物体
        
        Args:
            image_path: 原图路径
            mask_path: mask路径（白色=要移除的区域）
            output_path: 输出路径
            method: 修复方法
            inpaint_radius: 修复半径
        
        Returns:
            是否成功
        """
        method = method or self.method
        
        try:
            import cv2
            
            # 读取图片和mask
            img = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"[ObjectRemover] Cannot read image: {image_path}")
                return False
            if mask is None:
                print(f"[ObjectRemover] Cannot read mask: {mask_path}")
                return False
            
            # 确保mask是二值的
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 根据方法选择修复算法
            if method == "lama":
                result = self._inpaint_lama(img, mask)
            elif method == "cv2":
                result = self._inpaint_cv2(img, mask, inpaint_radius)
            else:
                # 自动选择：根据修复区域大小
                mask_area = np.sum(mask > 0)
                total_area = mask.shape[0] * mask.shape[1]
                ratio = mask_area / total_area
                
                if ratio > 0.1:  # 大面积用LaMa
                    result = self._inpaint_lama(img, mask)
                else:
                    result = self._inpaint_cv2(img, mask, inpaint_radius)
            
            if result is None:
                return False
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, result)
            print(f"[ObjectRemover] Saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[ObjectRemover] Error: {e}")
            return False
    
    def _inpaint_cv2(self, img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
        """使用OpenCV进行修复"""
        import cv2
        return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    
    def _inpaint_lama(self, img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """使用LaMa模型进行修复"""
        try:
            # 尝试加载LaMa
            if self._lama_model is None:
                self._load_lama()
            
            if self._lama_model is None:
                print("[ObjectRemover] LaMa not available, falling back to cv2")
                return self._inpaint_cv2(img, mask, 10)
            
            # LaMa推理
            import torch
            
            # 预处理
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            
            mask_tensor = torch.from_numpy(mask).float() / 255.0
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            
            # 移动到GPU
            device = next(self._lama_model.parameters()).device
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            # 推理
            with torch.no_grad():
                result = self._lama_model(img_tensor, mask_tensor)
            
            # 后处理
            result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result = (result * 255).astype(np.uint8)
            
            return result
            
        except Exception as e:
            print(f"[ObjectRemover] LaMa error: {e}")
            return self._inpaint_cv2(img, mask, 10)
    
    def _load_lama(self):
        """加载LaMa模型"""
        try:
            import torch
            
            # 模型路径
            model_path = os.path.join(os.path.dirname(__file__), "models", "lama", "big-lama.pt")
            
            if not os.path.exists(model_path):
                print(f"[ObjectRemover] LaMa model not found: {model_path}")
                print("[ObjectRemover] Download from: https://github.com/advimman/lama")
                return
            
            # 加载模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._lama_model = torch.jit.load(model_path, map_location=device)
            self._lama_model.eval()
            print(f"[ObjectRemover] LaMa model loaded on {device}")
            
        except Exception as e:
            print(f"[ObjectRemover] Failed to load LaMa: {e}")
    
    def remove_objects_by_detection(
        self,
        image_path: str,
        output_path: str,
        detect_type: str = "text",
        method: str = "auto",
    ) -> bool:
        """
        自动检测并移除物体
        
        detect_type: 检测类型
            - "text": 文字
            - "people": 人物
            - "all": 所有可检测物体
        """
        try:
            import cv2
            
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            # 生成mask
            mask = self._detect_objects(img, detect_type)
            
            if mask is None or np.sum(mask) == 0:
                print("[ObjectRemover] No objects detected")
                cv2.imwrite(output_path, img)
                return True
            
            # 修复
            return self.remove_object(image_path, None, output_path, method, mask=mask)
            
        except Exception as e:
            print(f"[ObjectRemover] Error: {e}")
            return False
    
    def _detect_objects(self, img: np.ndarray, detect_type: str) -> Optional[np.ndarray]:
        """检测物体生成mask"""
        import cv2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        if detect_type == "text":
            return self._detect_text(img)
        elif detect_type == "people":
            return self._detect_people(img)
        else:
            # 合并所有检测
            text_mask = self._detect_text(img)
            people_mask = self._detect_people(img)
            
            combined = np.zeros((height, width), dtype=np.uint8)
            if text_mask is not None:
                combined = cv2.bitwise_or(combined, text_mask)
            if people_mask is not None:
                combined = cv2.bitwise_or(combined, people_mask)
            
            return combined
    
    def _detect_text(self, img: np.ndarray) -> np.ndarray:
        """检测文字区域"""
        import cv2
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # 使用MSER检测文字
        mser = cv2.MSER_create()
        mser.setMinArea(30)
        mser.setMaxArea(int(height * width * 0.1))
        
        regions, _ = mser.detectRegions(gray)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            aspect_ratio = w / h if h > 0 else 0
            
            # 文字通常有一定的宽高比
            if 0.1 < aspect_ratio < 15 and w > 5 and h > 5:
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # 扩展mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask
    
    def _detect_people(self, img: np.ndarray) -> Optional[np.ndarray]:
        """检测人物"""
        try:
            import cv2
            
            # 使用HOG检测人物
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))
            
            mask = np.zeros(gray.shape, dtype=np.uint8)
            
            for (x, y, w, h) in boxes:
                # 扩展边界框
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(gray.shape[1], x + w + padding)
                y2 = min(gray.shape[0], y + h + padding)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            return mask
            
        except Exception as e:
            print(f"[ObjectRemover] People detection error: {e}")
            return None
    
    def interactive_remove(
        self,
        image_path: str,
        output_path: str,
        brush_mask: np.ndarray,
        method: str = "auto",
    ) -> bool:
        """
        交互式移除（用户画笔选择区域）
        
        brush_mask: 用户绘制的mask（白色=要移除的区域）
        """
        return self.remove_object(
            image_path, 
            None, 
            output_path, 
            method=method,
            mask=brush_mask
        )


class SimpleObjectRemover:
    """简化版杂物去除器"""
    
    def remove_object(
        self,
        image_path: str,
        mask: np.ndarray,
        output_path: str,
    ) -> bool:
        """使用OpenCV修复"""
        try:
            import cv2
            
            img = cv2.imread(image_path)
            if img is None:
                return False
            
            result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
            
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, result)
            return True
            
        except Exception as e:
            print(f"[SimpleObjectRemover] Error: {e}")
            return False
