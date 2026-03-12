"""
subject_segmenter.py
AI主体识别与分割模块
支持SAM2模型进行精确的主体分割
"""

import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


class SubjectSegmenter:
    """
    AI主体识别分割器
    使用SAM2进行精确的主体分割
    """
    
    def __init__(self, model_type: str = "sam2_hiera_small"):
        """
        初始化分割器
        
        model_type: 模型类型
            - sam2_hiera_tiny: 最快最小
            - sam2_hiera_small: 平衡选择（推荐）
            - sam2_hiera_base_plus: 效果更好
            - sam2_hiera_large: 最大最准
        """
        self.model_type = model_type
        self._predictor = None
        self._model_loaded = False
    
    def _ensure_model(self):
        """确保模型已加载"""
        if self._predictor is not None:
            return True
        
        try:
            # 尝试导入SAM2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            import torch
            
            # 检查GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[SubjectSegmenter] Using device: {device}")
            
            # 模型配置映射
            model_configs = {
                "sam2_hiera_tiny": ("sam2_hiera_tiny", "sam2_hiera_tiny.pt"),
                "sam2_hiera_small": ("sam2_hiera_small", "sam2_hiera_small.pt"),
                "sam2_hiera_base_plus": ("sam2_hiera_base_plus", "sam2_hiera_base_plus.pt"),
                "sam2_hiera_large": ("sam2_hiera_large", "sam2_hiera_large.pt"),
            }
            
            config_name, checkpoint_name = model_configs.get(
                self.model_type, 
                ("sam2_hiera_small", "sam2_hiera_small.pt")
            )
            
            # 模型路径
            sam2_dir = os.path.join(os.path.dirname(__file__), "models", "sam2")
            checkpoint_path = os.path.join(sam2_dir, checkpoint_name)
            
            # 检查模型文件是否存在
            if not os.path.exists(checkpoint_path):
                print(f"[SubjectSegmenter] Model not found at {checkpoint_path}")
                print(f"[SubjectSegmenter] Please download from: https://github.com/facebookresearch/sam2")
                return False
            
            # 构建模型
            sam2_model = build_sam2(config_name, checkpoint_path, device=device)
            self._predictor = SAM2ImagePredictor(sam2_model)
            self._model_loaded = True
            print(f"[SubjectSegmenter] Model loaded: {self.model_type}")
            return True
            
        except ImportError as e:
            print(f"[SubjectSegmenter] SAM2 not installed: {e}")
            print("[SubjectSegmenter] Install with: pip install sam2")
            return False
        except Exception as e:
            print(f"[SubjectSegmenter] Failed to load model: {e}")
            return False
    
    def segment_subject(
        self,
        image_path: str,
        output_path: str,
        output_type: str = "mask",
        point_coords: List[Tuple[int, int]] = None,
        box: Tuple[int, int, int, int] = None,
    ) -> bool:
        """
        分割主体
        
        Args:
            image_path: 输入图片路径
            output_path: 输出路径
            output_type: 输出类型
                - "mask": 黑白mask（白色主体）
                - "transparent": 透明背景PNG
                - "highlight": 高亮主体（红色半透明遮罩）
                - "extract": 提取主体到新背景
            point_coords: 提示点坐标列表 [(x,y), ...]
            box: 边界框提示
        
        Returns:
            是否成功
        """
        if not self._ensure_model():
            return self._fallback_segment(image_path, output_path, output_type)
        
        try:
            import cv2
            
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"[SubjectSegmenter] Cannot read image: {image_path}")
                return False
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 设置图片
            self._predictor.set_image(img_rgb)
            
            # 准备提示
            if point_coords is not None:
                points = np.array(point_coords)
                labels = np.ones(len(point_coords))  # 都是前景点
                masks, scores, _ = self._predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )
            elif box is not None:
                box_array = np.array(box)
                masks, scores, _ = self._predictor.predict(
                    box=box_array,
                    multimask_output=True,
                )
            else:
                # 自动模式：检测中心点作为提示
                h, w = img.shape[:2]
                center_points = np.array([
                    [w // 2, h // 2],
                    [w // 3, h // 3],
                    [2 * w // 3, h // 3],
                    [w // 2, 2 * h // 3],
                ])
                labels = np.ones(len(center_points))
                masks, scores, _ = self._predictor.predict(
                    point_coords=center_points,
                    point_labels=labels,
                    multimask_output=True,
                )
            
            # 选择最佳mask
            best_mask = masks[np.argmax(scores)]
            
            # 根据输出类型处理
            if output_type == "mask":
                result = (best_mask * 255).astype(np.uint8)
                cv2.imwrite(output_path, result)
            elif output_type == "transparent":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = (best_mask * 255).astype(np.uint8)
                cv2.imwrite(output_path, result)
            elif output_type == "highlight":
                overlay = img.copy()
                overlay[best_mask > 0] = [0, 0, 255]  # 红色
                result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
                cv2.imwrite(output_path, result)
            elif output_type == "extract":
                result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                result[:, :, 3] = (best_mask * 255).astype(np.uint8)
                cv2.imwrite(output_path, result)
            else:
                result = (best_mask * 255).astype(np.uint8)
                cv2.imwrite(output_path, result)
            
            print(f"[SubjectSegmenter] Saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"[SubjectSegmenter] Error: {e}")
            return self._fallback_segment(image_path, output_path, output_type)
    
    def _fallback_segment(self, image_path: str, output_path: str, output_type: str) -> bool:
        """备选方案：使用rembg进行分割"""
        try:
            print("[SubjectSegmenter] Using fallback (rembg)")
            from rembg import remove
            from PIL import Image
            
            img = Image.open(image_path).convert("RGBA")
            result = remove(img)
            
            if output_type == "mask":
                alpha = result.split()[3]
                mask = Image.new("RGB", alpha.size, (0, 0, 0))
                white = Image.new("RGB", alpha.size, (255, 255, 255))
                mask.paste(white, mask=alpha)
                mask.save(output_path)
            else:
                result.save(output_path)
            
            return True
        except Exception as e:
            print(f"[SubjectSegmenter] Fallback failed: {e}")
            return False
    
    def get_subject_mask(self, image_path: str) -> Optional[np.ndarray]:
        """获取主体mask数组"""
        if not self._ensure_model():
            return None
        
        try:
            import cv2
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self._predictor.set_image(img_rgb)
            
            h, w = img.shape[:2]
            center_points = np.array([[w // 2, h // 2]])
            labels = np.array([1])
            
            masks, scores, _ = self._predictor.predict(
                point_coords=center_points,
                point_labels=labels,
                multimask_output=True,
            )
            
            return masks[np.argmax(scores)]
        except Exception as e:
            print(f"[SubjectSegmenter] Error getting mask: {e}")
            return None
    
    def segment_multiple_subjects(
        self,
        image_path: str,
        output_dir: str,
        base_name: str = "subject",
    ) -> List[str]:
        """
        分割多个主体
        使用网格点检测多个主体
        """
        if not self._ensure_model():
            return []
        
        try:
            import cv2
            
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            self._predictor.set_image(img_rgb)
            
            # 网格点
            grid_points = []
            for y in range(h // 4, h, h // 4):
                for x in range(w // 4, w, w // 4):
                    grid_points.append([x, y])
            
            points = np.array(grid_points)
            labels = np.ones(len(grid_points))
            
            masks, scores, _ = self._predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
            
            # 合并所有mask
            combined_mask = np.any(masks, axis=0)
            
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{base_name}.png")
            
            result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = (combined_mask * 255).astype(np.uint8)
            cv2.imwrite(output_path, result)
            
            return [output_path]
            
        except Exception as e:
            print(f"[SubjectSegmenter] Error: {e}")
            return []


# 简化版分割器（无需SAM2）
class SimpleSegmenter:
    """简化版主体分割器，使用基础方法"""
    
    def segment_subject(
        self,
        image_path: str,
        output_path: str,
        output_type: str = "transparent",
    ) -> bool:
        """使用rembg进行分割"""
        try:
            from rembg import remove
            from PIL import Image
            
            img = Image.open(image_path).convert("RGBA")
            result = remove(img)
            
            if output_type == "mask":
                alpha = result.split()[3]
                mask = Image.new("RGB", alpha.size, (0, 0, 0))
                white = Image.new("RGB", alpha.size, (255, 255, 255))
                mask.paste(white, mask=alpha)
                mask.save(output_path, format="PNG")
            else:
                result.save(output_path, format="PNG")
            
            return True
        except Exception as e:
            print(f"[SimpleSegmenter] Error: {e}")
            return False
