"""
background_remover.py
使用 rembg (U²-Net) 自动去除图片背景，保留主体。
支持主体识别阈值控制，避免主体被误去除。
"""

import os
import numpy as np
from PIL import Image


def _get_session(model_name: str = "u2net"):
    """惰性加载 rembg session，避免启动时占用资源。"""
    try:
        from rembg import new_session
        return new_session(model_name)
    except ImportError:
        raise ImportError(
            "请先安装 rembg：pip install rembg\n"
            "首次运行时会自动下载模型文件（约 170MB）。"
        )


class BackgroundRemover:
    """
    AI 主体提取 / 背景去除器。
    
    支持主体识别阈值控制，防止主体被误去除。
    """

    SUPPORTED_MODELS = [
        "u2net",
        "u2net_human_seg",
        "isnet-general-use",
    ]

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            self._session = _get_session(self.model_name)

    def set_model(self, model_name: str):
        """切换模型时重置 session。"""
        if model_name != self.model_name:
            self.model_name = model_name
            self._session = None

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def remove_background(
        self,
        input_path: str,
        output_path: str,
        bg_color=None,
        # Alpha matting 参数
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        # 主体保留控制（新增）
        subject_strength: float = 1.0,
        edge_expand: int = 0,
        mask_smoothing: bool = False,
        min_subject_area: float = 0.0,
    ) -> bool:
        """
        对单张图片去除背景。

        参数
        ----
        input_path  : 源图片路径
        output_path : 输出路径（建议 .png 以保留透明度）
        bg_color    : None → 透明背景；(R,G,B) 或 (R,G,B,A) → 填充纯色背景
        
        alpha_matting : 是否启用 alpha matting（边缘更细腻，但更慢）
        alpha_matting_foreground_threshold : 前景阈值（0-255，越高保留越多前景）
        alpha_matting_background_threshold : 背景阈值（0-255，越低保留越多背景）
        alpha_matting_erode_size : 腐蚀大小
        
        subject_strength : 主体保留强度（0.5-2.0，>1 保留更多主体，<1 去除更多背景）
        edge_expand : 边缘扩展像素数（扩大主体区域，防止边缘被误删）
        mask_smoothing : 是否平滑 mask 边缘
        min_subject_area : 最小主体面积比例（0-1，小于此比例时保留原图）
        """
        try:
            from rembg import remove as rembg_remove

            self._ensure_session()

            img = Image.open(input_path).convert("RGBA")
            original_size = img.size

            kwargs = dict(
                session=self._session,
                alpha_matting=alpha_matting,
            )
            if alpha_matting:
                kwargs["alpha_matting_foreground_threshold"] = alpha_matting_foreground_threshold
                kwargs["alpha_matting_background_threshold"] = alpha_matting_background_threshold
                kwargs["alpha_matting_erode_size"] = alpha_matting_erode_size

            # 获取原始 mask
            result: Image.Image = rembg_remove(img, **kwargs)
            
            # 提取 alpha 通道作为 mask
            mask = result.split()[3]
            mask_array = np.array(mask)
            
            # 应用主体保留强度
            if subject_strength != 1.0:
                mask_array = self._adjust_mask_strength(mask_array, subject_strength)
            
            # 边缘扩展
            if edge_expand > 0:
                mask_array = self._expand_mask(mask_array, edge_expand)
            
            # 平滑处理
            if mask_smoothing:
                mask_array = self._smooth_mask(mask_array)
            
            # 检查主体面积
            if min_subject_area > 0:
                subject_ratio = np.sum(mask_array > 0) / mask_array.size
                if subject_ratio < min_subject_area:
                    print(f"[INFO] 主体面积 {subject_ratio*100:.1f}% < {min_subject_area*100:.1f}%，保留原图")
                    img.save(output_path, format="PNG")
                    return True
            
            # 应用调整后的 mask
            mask = Image.fromarray(mask_array)
            result = img.copy()
            result.putalpha(mask)

            # 若指定了背景色，合成到纯色画布上
            if bg_color is not None:
                if len(bg_color) == 3:
                    bg_color = (*bg_color, 255)
                bg = Image.new("RGBA", result.size, bg_color)
                bg.paste(result, mask=result.split()[3])
                result = bg.convert("RGB")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            result.save(output_path, format="PNG")
            return True

        except Exception as e:
            print(f"[BackgroundRemover] 处理失败 {input_path}: {e}")
            raise

    def _adjust_mask_strength(self, mask: np.ndarray, strength: float) -> np.ndarray:
        """
        调整 mask 强度。
        strength > 1: 保留更多主体（降低阈值）
        strength < 1: 去除更多背景（提高阈值）
        """
        if strength == 1.0:
            return mask
        
        # 归一化到 0-1
        mask_norm = mask.astype(np.float32) / 255.0
        
        if strength > 1.0:
            # 保留更多主体：降低判断阈值
            # strength=1.5 时，原来0.6的置信度变成0.9，更多像素被保留
            mask_norm = np.power(mask_norm, 1.0 / strength)
        else:
            # 去除更多背景：提高判断阈值
            # strength=0.7 时，需要更高的置信度才保留
            mask_norm = np.power(mask_norm, 1.0 / strength)
        
        # 转回 0-255
        return (mask_norm * 255).astype(np.uint8)

    def _expand_mask(self, mask: np.ndarray, pixels: int) -> np.ndarray:
        """扩展 mask 边缘，防止主体边缘被误删。"""
        import cv2
        
        kernel = np.ones((pixels * 2 + 1, pixels * 2 + 1), np.uint8)
        expanded = cv2.dilate(mask, kernel, iterations=1)
        return expanded

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """平滑 mask 边缘。"""
        import cv2
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        return blurred

    def remove_background_batch(
        self,
        input_paths: list,
        output_dir: str,
        base_name: str = "nobg",
        bg_color=None,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        subject_strength: float = 1.0,
        edge_expand: int = 0,
        mask_smoothing: bool = False,
        min_subject_area: float = 0.0,
        progress_callback=None,
    ) -> list:
        """
        批量去除背景。
        """
        os.makedirs(output_dir, exist_ok=True)
        self._ensure_session()

        output_paths = []
        total = len(input_paths)

        for idx, input_path in enumerate(input_paths):
            filename = os.path.basename(input_path)
            if progress_callback:
                progress_callback(idx + 1, total, filename)

            name_no_ext = os.path.splitext(filename)[0]
            suffix = f"_{idx + 1}" if len(input_paths) > 1 else ""
            out_name = f"{base_name}{suffix}.png" if base_name else f"{name_no_ext}_nobg.png"
            output_path = os.path.join(output_dir, out_name)

            self.remove_background(
                input_path,
                output_path,
                bg_color=bg_color,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                subject_strength=subject_strength,
                edge_expand=edge_expand,
                mask_smoothing=mask_smoothing,
                min_subject_area=min_subject_area,
            )
            output_paths.append(output_path)

        return output_paths

    def preview_mask(
        self, 
        input_path: str, 
        output_path: str,
        subject_strength: float = 1.0,
        edge_expand: int = 0,
        mask_smoothing: bool = False,
    ) -> bool:
        """
        生成主体 mask 预览（白色主体 + 黑色背景），用于检验检测效果。
        """
        try:
            from rembg import remove as rembg_remove

            self._ensure_session()
            img = Image.open(input_path).convert("RGBA")
            result: Image.Image = rembg_remove(img, session=self._session)

            alpha = result.split()[3]
            mask_array = np.array(alpha)
            
            # 应用主体保留强度
            if subject_strength != 1.0:
                mask_array = self._adjust_mask_strength(mask_array, subject_strength)
            
            # 边缘扩展
            if edge_expand > 0:
                mask_array = self._expand_mask(mask_array, edge_expand)
            
            # 平滑处理
            if mask_smoothing:
                mask_array = self._smooth_mask(mask_array)
            
            mask = Image.fromarray(mask_array)
            preview = Image.new("RGB", mask.size, (0, 0, 0))
            white = Image.new("RGB", mask.size, (255, 255, 255))
            preview.paste(white, mask=mask)

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            preview.save(output_path, format="PNG")
            return True

        except Exception as e:
            print(f"[BackgroundRemover] 预览失败 {input_path}: {e}")
            raise

    def get_subject_ratio(self, input_path: str) -> float:
        """
        获取主体占图像的比例（0-1）。
        用于判断是否需要调整阈值。
        """
        try:
            from rembg import remove as rembg_remove

            self._ensure_session()
            img = Image.open(input_path).convert("RGBA")
            result: Image.Image = rembg_remove(img, session=self._session)

            alpha = result.split()[3]
            mask_array = np.array(alpha)
            
            return np.sum(mask_array > 128) / mask_array.size

        except Exception as e:
            print(f"[BackgroundRemover] 获取主体比例失败: {e}")
            return 0.0
