"""
background_remover.py
使用 rembg (U²-Net) 自动去除图片背景，保留主体。
输出为带透明通道的 PNG，或以指定颜色填充背景。
"""

import os
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

    参数
    ----
    model_name : str
        rembg 支持的模型名称。
        - "u2net"      通用主体（默认，推荐）
        - "u2net_human_seg"  人像专用，效果更精细
        - "isnet-general-use"  更新的通用模型
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
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
    ) -> bool:
        """
        对单张图片去除背景。

        参数
        ----
        input_path  : 源图片路径
        output_path : 输出路径（建议 .png 以保留透明度）
        bg_color    : None → 透明背景；(R,G,B) 或 (R,G,B,A) → 填充纯色背景
        alpha_matting : 是否启用 alpha matting（边缘更细腻，但更慢）
        """
        try:
            from rembg import remove as rembg_remove

            self._ensure_session()

            img = Image.open(input_path).convert("RGBA")

            kwargs = dict(
                session=self._session,
                alpha_matting=alpha_matting,
            )
            if alpha_matting:
                kwargs["alpha_matting_foreground_threshold"] = alpha_matting_foreground_threshold
                kwargs["alpha_matting_background_threshold"] = alpha_matting_background_threshold
                kwargs["alpha_matting_erode_size"] = alpha_matting_erode_size

            result: Image.Image = rembg_remove(img, **kwargs)

            # 若指定了背景色，合成到纯色画布上
            if bg_color is not None:
                if len(bg_color) == 3:
                    bg_color = (*bg_color, 255)
                bg = Image.new("RGBA", result.size, bg_color)
                bg.paste(result, mask=result.split()[3])
                result = bg.convert("RGB")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            result.save(output_path)
            return True

        except Exception as e:
            print(f"[BackgroundRemover] 处理失败 {input_path}: {e}")
            raise

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
        progress_callback=None,
    ) -> list:
        """
        批量去除背景。

        返回成功输出的路径列表。
        progress_callback(current, total, filename) 可用于更新进度。
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
            )
            output_paths.append(output_path)

        return output_paths

    def preview_mask(self, input_path: str, output_path: str) -> bool:
        """
        生成主体 mask 预览（白色主体 + 黑色背景），用于检验检测效果。
        """
        try:
            from rembg import remove as rembg_remove

            self._ensure_session()
            img = Image.open(input_path).convert("RGBA")
            result: Image.Image = rembg_remove(img, session=self._session)

            alpha = result.split()[3]
            mask = Image.new("RGB", alpha.size, (0, 0, 0))
            white = Image.new("RGB", alpha.size, (255, 255, 255))
            mask.paste(white, mask=alpha)

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            mask.save(output_path)
            return True

        except Exception as e:
            print(f"[BackgroundRemover] 预览失败 {input_path}: {e}")
            raise
