import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading
from typing import List
import queue

from image_similarity import ImageSimilarityDetector
from image_stitch import stitch_images_horizontal, stitch_images_vertical, stitch_images_grid
from image_split import split_image_horizontal, split_image_vertical, split_image_grid, auto_split_image
from smart_split import SmartSplitDetector
from text_remover import TextRemover


class ImageToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("图像工具箱 v4.0")
        self.root.geometry("950x850")
        
        self.detector = None
        self.smart_detector = SmartSplitDetector()
        self.text_remover = TextRemover()
        self.image_paths = []
        self.split_batch_paths = []
        self.text_remove_paths = []
        self.stitched_images = []
        
        self.progress_queue = queue.Queue()
        
        self.create_widgets()
        self.check_progress_queue()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky="nsew")
        
        stitch_frame = ttk.Frame(notebook, padding="10")
        split_frame = ttk.Frame(notebook, padding="10")
        text_frame = ttk.Frame(notebook, padding="10")
        
        notebook.add(stitch_frame, text="图像拼接")
        notebook.add(split_frame, text="图像拆分")
        notebook.add(text_frame, text="文字去除")
        
        self.create_stitch_tab(stitch_frame)
        self.create_split_tab(split_frame)
        self.create_text_remove_tab(text_frame)
        
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, sticky="ew", pady=(5, 0))
    
    def create_stitch_tab(self, parent):
        input_frame = ttk.LabelFrame(parent, text="输入图片", padding="10")
        input_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="添加图片", command=self.add_images).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_images).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="AI自动分组", command=self.auto_group_images).grid(row=0, column=2, padx=5)
        
        list_frame = ttk.Frame(input_frame)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")
        
        self.image_listbox = tk.Listbox(list_frame, height=8, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.image_listbox.yview)
        self.image_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        options_frame = ttk.LabelFrame(parent, text="拼接选项", padding="10")
        options_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        ttk.Label(options_frame, text="拼接方式:").grid(row=0, column=0, sticky=tk.W)
        self.stitch_mode = tk.StringVar(value="horizontal")
        mode_combo = ttk.Combobox(options_frame, textvariable=self.stitch_mode, 
                                   values=["horizontal", "vertical", "grid"], state="readonly", width=15)
        mode_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(options_frame, text="网格列数(仅grid模式):").grid(row=0, column=2, padx=(20, 5))
        self.grid_cols = tk.IntVar(value=2)
        ttk.Spinbox(options_frame, from_=1, to=10, textvariable=self.grid_cols, width=5).grid(row=0, column=3)
        
        output_frame = ttk.LabelFrame(parent, text="输出设置", padding="10")
        output_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W)
        self.output_dir = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="浏览", command=self.browse_output_dir).grid(row=0, column=2)
        
        ttk.Label(output_frame, text="输出文件名:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.output_filename = tk.StringVar(value="stitched_image")
        ttk.Entry(output_frame, textvariable=self.output_filename, width=50).grid(row=1, column=1, padx=5, pady=(10, 0))
        
        ttk.Label(output_frame, text="相似度阈值:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.similarity_threshold = tk.DoubleVar(value=0.85)
        ttk.Scale(output_frame, from_=0.5, to=1.0, variable=self.similarity_threshold, 
                  orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        ttk.Label(output_frame, textvariable=self.similarity_threshold).grid(row=2, column=2, pady=(10, 0))
        
        ttk.Button(parent, text="开始拼接", command=self.start_stitch).grid(row=3, column=0, pady=20)
    
    def create_split_tab(self, parent):
        input_frame = ttk.LabelFrame(parent, text="输入图片", padding="10")
        input_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="选择单张图片", command=self.select_split_image).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="批量选择图片", command=self.select_batch_split_images).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_split_image).grid(row=0, column=2, padx=5)
        
        list_frame = ttk.Frame(input_frame)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")
        
        self.split_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.split_listbox.yview)
        self.split_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.split_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        options_frame = ttk.LabelFrame(parent, text="拆分选项", padding="10")
        options_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        ttk.Label(options_frame, text="拆分模式:").grid(row=0, column=0, sticky=tk.W)
        self.split_mode = tk.StringVar(value="ai_smart")
        mode_combo = ttk.Combobox(options_frame, textvariable=self.split_mode,
                                   values=["ai_smart", "auto", "horizontal", "vertical", "grid"], state="readonly", width=15)
        mode_combo.grid(row=0, column=1, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_split_mode_change)
        
        self.auto_parts_label = ttk.Label(options_frame, text="拆分数量(非AI模式):")
        self.auto_parts_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.split_parts = tk.IntVar(value=3)
        self.auto_parts_spinbox = ttk.Spinbox(options_frame, from_=2, to=20, textvariable=self.split_parts, width=5)
        self.auto_parts_spinbox.grid(row=1, column=1, pady=(10, 0), sticky=tk.W)
        
        self.grid_label = ttk.Label(options_frame, text="网格行数(grid模式):")
        self.grid_label.grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.split_rows = tk.IntVar(value=2)
        self.grid_row_spinbox = ttk.Spinbox(options_frame, from_=1, to=10, textvariable=self.split_rows, width=5)
        self.grid_row_spinbox.grid(row=2, column=1, pady=(10, 0), sticky=tk.W)
        
        self.grid_col_label = ttk.Label(options_frame, text="网格列数(grid模式):")
        self.grid_col_label.grid(row=2, column=2, padx=(20, 0), sticky=tk.W, pady=(10, 0))
        self.split_cols = tk.IntVar(value=2)
        self.grid_col_spinbox = ttk.Spinbox(options_frame, from_=1, to=10, textvariable=self.split_cols, width=5)
        self.grid_col_spinbox.grid(row=2, column=3, pady=(10, 0))
        
        self.on_split_mode_change()
        
        output_frame = ttk.LabelFrame(parent, text="输出设置", padding="10")
        output_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W)
        self.split_output_dir = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.split_output_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="浏览", command=self.browse_split_output_dir).grid(row=0, column=2)
        
        ttk.Label(output_frame, text="基础文件名:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.split_base_name = tk.StringVar(value="split_image")
        ttk.Entry(output_frame, textvariable=self.split_base_name, width=50).grid(row=1, column=1, padx=5, pady=(10, 0))
        
        ttk.Button(parent, text="开始拆分", command=self.start_split).grid(row=3, column=0, pady=20)
    
    def create_text_remove_tab(self, parent):
        input_frame = ttk.LabelFrame(parent, text="输入图片", padding="10")
        input_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Button(btn_frame, text="选择单张图片", command=self.select_text_remove_image).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="批量选择图片", command=self.select_batch_text_remove_images).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="清空列表", command=self.clear_text_remove_image).grid(row=0, column=2, padx=5)
        
        list_frame = ttk.Frame(input_frame)
        list_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky="nsew")
        
        self.text_listbox = tk.Listbox(list_frame, height=6, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.text_listbox.yview)
        self.text_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.text_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        options_frame = ttk.LabelFrame(parent, text="文字去除选项", padding="10")
        options_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        ttk.Label(options_frame, text="检测方法:").grid(row=0, column=0, sticky=tk.W)
        self.text_detect_method = tk.StringVar(value="auto")
        detect_combo = ttk.Combobox(options_frame, textvariable=self.text_detect_method,
                                     values=["auto", "mser", "edge"], state="readonly", width=15)
        detect_combo.grid(row=0, column=1, padx=5)
        ttk.Label(options_frame, text="(auto自动选择最佳方法)").grid(row=0, column=2, padx=5, sticky=tk.W)
        
        ttk.Label(options_frame, text="膨胀大小:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.text_dilate_size = tk.IntVar(value=5)
        ttk.Spinbox(options_frame, from_=1, to=20, textvariable=self.text_dilate_size, width=5).grid(row=1, column=1, pady=(10, 0), sticky=tk.W)
        ttk.Label(options_frame, text="(扩展文字区域，防止遗漏边缘)").grid(row=1, column=2, padx=5, pady=(10, 0), sticky=tk.W)
        
        ttk.Label(options_frame, text="修复半径:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.text_inpaint_radius = tk.IntVar(value=5)
        ttk.Spinbox(options_frame, from_=1, to=30, textvariable=self.text_inpaint_radius, width=5).grid(row=2, column=1, pady=(10, 0), sticky=tk.W)
        ttk.Label(options_frame, text="(修复区域半径，越大越平滑)").grid(row=2, column=2, padx=5, pady=(10, 0), sticky=tk.W)
        
        output_frame = ttk.LabelFrame(parent, text="输出设置", padding="10")
        output_frame.grid(row=2, column=0, sticky="ew", pady=5)
        
        ttk.Label(output_frame, text="输出目录:").grid(row=0, column=0, sticky=tk.W)
        self.text_output_dir = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.text_output_dir, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(output_frame, text="浏览", command=self.browse_text_output_dir).grid(row=0, column=2)
        
        ttk.Label(output_frame, text="基础文件名:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.text_base_name = tk.StringVar(value="text_removed")
        ttk.Entry(output_frame, textvariable=self.text_base_name, width=50).grid(row=1, column=1, padx=5, pady=(10, 0))
        
        btn_frame2 = ttk.Frame(parent)
        btn_frame2.grid(row=3, column=0, pady=20)
        
        ttk.Button(btn_frame2, text="预览文字检测", command=self.preview_text_detection).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frame2, text="开始去除文字", command=self.start_text_remove).grid(row=0, column=1, padx=10)
    
    def on_split_mode_change(self, event=None):
        mode = self.split_mode.get()
        is_ai = mode == "ai_smart"
        is_grid = mode == "grid"
        
        state_parts = "disabled" if is_ai else "normal"
        state_grid = "disabled" if not is_grid else "normal"
        
        self.auto_parts_spinbox.config(state=state_parts)
        self.grid_row_spinbox.config(state=state_grid)
        self.grid_col_spinbox.config(state=state_grid)
    
    def add_images(self):
        files = filedialog.askopenfilenames(
            title="选择图片文件",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if files:
            for f in files:
                if f not in self.image_paths:
                    self.image_paths.append(f)
                    self.image_listbox.insert(tk.END, os.path.basename(f))
    
    def clear_images(self):
        self.image_paths.clear()
        self.image_listbox.delete(0, tk.END)
    
    def auto_group_images(self):
        if not self.image_paths:
            messagebox.showwarning("警告", "请先添加图片！")
            return
        
        self.status_var.set("正在初始化AI模型...")
        self.progress.start()
        
        def group_thread():
            try:
                if self.detector is None:
                    self.detector = ImageSimilarityDetector()
                
                self.progress_queue.put(("status", "正在分析图片相似度..."))
                
                threshold = self.similarity_threshold.get()
                groups = self.detector.group_similar_images(
                    self.image_paths,
                    similarity_threshold=threshold,
                    min_samples=1
                )
                
                result_msg = f"检测到 {len(groups)} 个图片组\n\n"
                for group_id, paths in sorted(groups.items()):
                    result_msg += f"组 {group_id + 1}:\n"
                    for path in paths:
                        result_msg += f"  - {os.path.basename(path)}\n"
                    result_msg += "\n"
                
                self.progress_queue.put(("result", result_msg))
                self.progress_queue.put(("groups", groups))
                
            except Exception as e:
                self.progress_queue.put(("error", str(e)))
        
        threading.Thread(target=group_thread, daemon=True).start()
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_dir.set(directory)
    
    def start_stitch(self):
        if not self.image_paths:
            messagebox.showwarning("警告", "请先添加图片！")
            return
        
        output_dir = self.output_dir.get()
        if not output_dir:
            messagebox.showwarning("警告", "请选择输出目录！")
            return
        
        filename = self.output_filename.get()
        if not filename:
            messagebox.showwarning("警告", "请输入输出文件名！")
            return
        
        output_path = os.path.join(output_dir, f"{filename}.png")
        
        self.status_var.set("正在拼接图片...")
        self.progress.start()
        
        def stitch_thread():
            try:
                mode = self.stitch_mode.get()
                
                if mode == "horizontal":
                    stitch_images_horizontal(self.image_paths, output_path)
                elif mode == "vertical":
                    stitch_images_vertical(self.image_paths, output_path)
                elif mode == "grid":
                    cols = self.grid_cols.get()
                    stitch_images_grid(self.image_paths, output_path, cols)
                
                self.progress_queue.put(("success", f"拼接完成！\n输出文件: {output_path}"))
                
            except Exception as e:
                self.progress_queue.put(("error", str(e)))
        
        threading.Thread(target=stitch_thread, daemon=True).start()
    
    def select_split_image(self):
        file = filedialog.askopenfilename(
            title="选择要拆分的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if file:
            self.split_batch_paths = [file]
            self.split_listbox.delete(0, tk.END)
            self.split_listbox.insert(tk.END, os.path.basename(file))
    
    def select_batch_split_images(self):
        files = filedialog.askopenfilenames(
            title="批量选择要拆分的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if files:
            self.split_batch_paths = list(files)
            self.split_listbox.delete(0, tk.END)
            for f in files:
                self.split_listbox.insert(tk.END, os.path.basename(f))
    
    def clear_split_image(self):
        self.split_batch_paths.clear()
        self.split_listbox.delete(0, tk.END)
    
    def browse_split_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.split_output_dir.set(directory)
    
    def start_split(self):
        if not self.split_batch_paths:
            messagebox.showwarning("警告", "请先选择要拆分的图片！")
            return
        
        output_dir = self.split_output_dir.get()
        if not output_dir:
            messagebox.showwarning("警告", "请选择输出目录！")
            return
        
        base_name = self.split_base_name.get()
        if not base_name:
            messagebox.showwarning("警告", "请输入基础文件名！")
            return
        
        self.status_var.set("正在拆分图片...")
        self.progress.start()
        
        def split_thread():
            try:
                mode = self.split_mode.get()
                total_outputs = []
                error_msgs = []
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    self.progress_queue.put(("status", f"创建输出目录: {output_dir}"))
                
                for img_idx, image_path in enumerate(self.split_batch_paths):
                    if not os.path.exists(image_path):
                        error_msgs.append(f"图片不存在: {image_path}")
                        continue
                    
                    self.progress_queue.put(("status", f"正在处理: {os.path.basename(image_path)}"))
                    
                    img_base = f"{base_name}_{img_idx + 1}" if len(self.split_batch_paths) > 1 else base_name
                    
                    output_paths = []
                    try:
                        if mode == "ai_smart":
                            output_paths = self.smart_detector.smart_split(image_path, output_dir, img_base)
                        elif mode == "auto":
                            n_parts = self.split_parts.get()
                            output_paths = auto_split_image(image_path, n_parts, output_dir, img_base)
                        elif mode == "horizontal":
                            n_parts = self.split_parts.get()
                            output_paths = split_image_horizontal(image_path, n_parts, output_dir, img_base)
                        elif mode == "vertical":
                            n_parts = self.split_parts.get()
                            output_paths = split_image_vertical(image_path, n_parts, output_dir, img_base)
                        elif mode == "grid":
                            rows = self.split_rows.get()
                            cols = self.split_cols.get()
                            output_paths = split_image_grid(image_path, rows, cols, output_dir, img_base)
                    except Exception as e:
                        error_msgs.append(f"{os.path.basename(image_path)}: {str(e)}")
                        continue
                    
                    if output_paths:
                        total_outputs.extend(output_paths)
                    else:
                        error_msgs.append(f"{os.path.basename(image_path)}: 未检测到可拆分区域")
                
                if error_msgs:
                    self.progress_queue.put(("error", f"部分失败:\n" + "\n".join(error_msgs)))
                elif total_outputs:
                    self.progress_queue.put(("success", f"拆分完成！\n处理 {len(self.split_batch_paths)} 张图片，生成 {len(total_outputs)} 张图片\n输出目录: {output_dir}"))
                else:
                    self.progress_queue.put(("error", "未能生成任何输出文件"))
                
            except Exception as e:
                import traceback
                self.progress_queue.put(("error", f"错误:\n{traceback.format_exc()}"))
        
        threading.Thread(target=split_thread, daemon=True).start()
    
    def select_text_remove_image(self):
        file = filedialog.askopenfilename(
            title="选择要去除文字的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if file:
            self.text_remove_paths = [file]
            self.text_listbox.delete(0, tk.END)
            self.text_listbox.insert(tk.END, os.path.basename(file))
    
    def select_batch_text_remove_images(self):
        files = filedialog.askopenfilenames(
            title="批量选择要去除文字的图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if files:
            self.text_remove_paths = list(files)
            self.text_listbox.delete(0, tk.END)
            for f in files:
                self.text_listbox.insert(tk.END, os.path.basename(f))
    
    def clear_text_remove_image(self):
        self.text_remove_paths.clear()
        self.text_listbox.delete(0, tk.END)
    
    def browse_text_output_dir(self):
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.text_output_dir.set(directory)
    
    def preview_text_detection(self):
        if not self.text_remove_paths:
            messagebox.showwarning("警告", "请先选择图片！")
            return
        
        output_dir = self.text_output_dir.get()
        if not output_dir:
            messagebox.showwarning("警告", "请选择输出目录！")
            return
        
        self.status_var.set("正在生成预览...")
        self.progress.start()
        
        def preview_thread():
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                detect_method = self.text_detect_method.get()
                
                for img_path in self.text_remove_paths:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
                    
                    success = self.text_remover.preview_text_detection(img_path, preview_path, detect_method)
                    if success:
                        self.progress_queue.put(("status", f"预览已保存: {preview_path}"))
                
                self.progress_queue.put(("success", f"预览生成完成！\n输出目录: {output_dir}\n红色区域为检测到的文字区域"))
                
            except Exception as e:
                import traceback
                self.progress_queue.put(("error", f"错误:\n{traceback.format_exc()}"))
        
        threading.Thread(target=preview_thread, daemon=True).start()
    
    def start_text_remove(self):
        if not self.text_remove_paths:
            messagebox.showwarning("警告", "请先选择图片！")
            return
        
        output_dir = self.text_output_dir.get()
        if not output_dir:
            messagebox.showwarning("警告", "请选择输出目录！")
            return
        
        self.status_var.set("正在去除文字...")
        self.progress.start()
        
        def remove_thread():
            try:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                detect_method = self.text_detect_method.get()
                dilate_size = self.text_dilate_size.get()
                inpaint_radius = self.text_inpaint_radius.get()
                base_name = self.text_base_name.get()
                
                success_count = 0
                error_msgs = []
                
                for img_idx, img_path in enumerate(self.text_remove_paths):
                    self.progress_queue.put(("status", f"正在处理: {os.path.basename(img_path)}"))
                    
                    img_base = f"{base_name}_{img_idx + 1}" if len(self.text_remove_paths) > 1 else base_name
                    output_path = os.path.join(output_dir, f"{img_base}.png")
                    
                    try:
                        success = self.text_remover.remove_text(
                            img_path, output_path,
                            dilate_size=dilate_size,
                            inpaint_radius=inpaint_radius,
                            detection_method=detect_method
                        )
                        if success:
                            success_count += 1
                        else:
                            error_msgs.append(f"{os.path.basename(img_path)}: 处理失败")
                    except Exception as e:
                        error_msgs.append(f"{os.path.basename(img_path)}: {str(e)}")
                
                if error_msgs:
                    self.progress_queue.put(("error", f"部分失败:\n" + "\n".join(error_msgs)))
                elif success_count > 0:
                    self.progress_queue.put(("success", f"文字去除完成！\n成功处理 {success_count} 张图片\n输出目录: {output_dir}"))
                else:
                    self.progress_queue.put(("error", "处理失败，未生成任何输出文件"))
                
            except Exception as e:
                import traceback
                self.progress_queue.put(("error", f"错误:\n{traceback.format_exc()}"))
        
        threading.Thread(target=remove_thread, daemon=True).start()
    
    def check_progress_queue(self):
        try:
            while True:
                msg_type, msg_data = self.progress_queue.get_nowait()
                
                self.progress.stop()
                
                if msg_type == "status":
                    self.status_var.set(msg_data)
                elif msg_type == "success":
                    self.status_var.set("完成")
                    messagebox.showinfo("成功", msg_data)
                elif msg_type == "error":
                    self.status_var.set("错误")
                    messagebox.showerror("错误", f"操作失败:\n{msg_data}")
                elif msg_type == "result":
                    self.status_var.set("完成")
                    messagebox.showinfo("AI分组结果", msg_data)
                elif msg_type == "groups":
                    pass
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_progress_queue)


def main():
    root = tk.Tk()
    app = ImageToolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
