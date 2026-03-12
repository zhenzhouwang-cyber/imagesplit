"""
style_config.py
界面样式配置
定义现代化的配色方案和组件样式
"""

import tkinter as tk
from tkinter import ttk

# ═══════════════════════════════════════════════════════════════════════
# 配色方案
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    # 主色调
    'primary': '#3B82F6',       # 蓝色 - 主按钮、选中状态
    'primary_hover': '#2563EB', # 蓝色悬停
    'primary_dark': '#1D4ED8',  # 蓝色按下
    
    # 辅助色
    'success': '#10B981',       # 绿色 - 成功/确认
    'warning': '#F59E0B',       # 橙色 - 提醒
    'danger': '#EF4444',        # 红色 - 删除/清除
    
    # 中性色
    'text_primary': '#1F2937',    # 深灰 - 主文字
    'text_secondary': '#6B7280',  # 中灰 - 次文字
    'text_disabled': '#9CA3AF',   # 浅灰 - 禁用文字
    'text_white': '#FFFFFF',      # 白色 - 按钮文字
    
    # 背景色
    'bg_primary': '#FFFFFF',      # 白色 - 主背景
    'bg_secondary': '#F9FAFB',    # 浅灰 - 次背景
    'bg_hover': '#F3F4F6',        # 悬停背景
    'bg_selected': '#EFF6FF',     # 选中背景
    
    # 边框色
    'border': '#E5E7EB',          # 边框
    'border_focus': '#3B82F6',    # 聚焦边框
    
    # 标签页颜色
    'tab_bg': '#F3F4F6',
    'tab_selected': '#3B82F6',
}

# ═══════════════════════════════════════════════════════════════════════
# 样式配置
# ═══════════════════════════════════════════════════════════════════════

FONTS = {
    'title': ('Microsoft YaHei UI', 14, 'bold'),
    'subtitle': ('Microsoft YaHei UI', 11, 'bold'),
    'body': ('Microsoft YaHei UI', 10),
    'small': ('Microsoft YaHei UI', 9),
    'button': ('Microsoft YaHei UI', 10, 'bold'),
}

BUTTON_STYLE = {
    'padding': (16, 8),
    'relief': 'flat',
    'cursor': 'hand2',
}


def configure_styles():
    """配置全局样式"""
    style = ttk.Style()
    
    # 使用clam主题作为基础
    try:
        style.theme_use('clam')
    except:
        pass
    
    # ─────────────────────────────────────────────────────────────────
    # Notebook (标签页容器)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TNotebook', background=COLORS['bg_primary'])
    style.configure('TNotebook.Tab',
        padding=[20, 10],
        background=COLORS['tab_bg'],
        foreground=COLORS['text_secondary'],
        font=FONTS['body'])
    style.map('TNotebook.Tab',
        background=[('selected', COLORS['tab_selected'])],
        foreground=[('selected', COLORS['text_white'])],
        expand=[('selected', [1, 1, 1, 0])])
    
    # ─────────────────────────────────────────────────────────────────
    # Frame (框架)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TFrame', background=COLORS['bg_primary'])
    style.configure('Card.TFrame', 
        background=COLORS['bg_primary'],
        relief='solid',
        borderwidth=1)
    style.configure('Section.TLabelframe', 
        background=COLORS['bg_primary'],
        relief='solid',
        borderwidth=1,
        bordercolor=COLORS['border'])
    style.configure('Section.TLabelframe.Label',
        background=COLORS['bg_primary'],
        foreground=COLORS['text_primary'],
        font=FONTS['subtitle'])
    
    # ─────────────────────────────────────────────────────────────────
    # Buttons (按钮)
    # ─────────────────────────────────────────────────────────────────
    
    # 主按钮 - 蓝色填充
    style.configure('Primary.TButton',
        padding=(24, 12),
        background=COLORS['primary'],
        foreground=COLORS['text_white'],
        font=FONTS['button'],
        relief='flat',
        borderwidth=0)
    style.map('Primary.TButton',
        background=[('active', COLORS['primary_hover']),
                   ('pressed', COLORS['primary_dark'])],
        foreground=[('active', COLORS['text_white']),
                   ('pressed', COLORS['text_white'])])
    
    # 次按钮 - 描边样式
    style.configure('Secondary.TButton',
        padding=(16, 8),
        background=COLORS['bg_primary'],
        foreground=COLORS['primary'],
        font=FONTS['body'],
        relief='solid',
        borderwidth=1)
    style.map('Secondary.TButton',
        background=[('active', COLORS['bg_selected'])])
    
    # 危险按钮 - 红色
    style.configure('Danger.TButton',
        padding=(16, 8),
        background=COLORS['danger'],
        foreground=COLORS['text_white'],
        font=FONTS['body'],
        relief='flat')
    style.map('Danger.TButton',
        background=[('active', '#DC2626')])
    
    # 成功按钮 - 绿色
    style.configure('Success.TButton',
        padding=(16, 8),
        background=COLORS['success'],
        foreground=COLORS['text_white'],
        font=FONTS['body'],
        relief='flat')
    
    # 小按钮
    style.configure('Small.TButton',
        padding=(12, 6),
        font=FONTS['small'])
    
    # 图标按钮
    style.configure('Icon.TButton',
        padding=(12, 12),
        font=('Segoe UI Symbol', 12))
    
    # 卡片按钮 - 用于选择模式
    style.configure('CardButton.TButton',
        padding=(20, 15),
        background=COLORS['bg_secondary'],
        foreground=COLORS['text_primary'],
        font=FONTS['body'],
        relief='solid',
        borderwidth=1)
    style.map('CardButton.TButton',
        background=[('active', COLORS['bg_selected']),
                   ('selected', COLORS['primary'])],
        foreground=[('selected', COLORS['text_white'])])
    
    # ─────────────────────────────────────────────────────────────────
    # Labels (标签)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TLabel',
        background=COLORS['bg_primary'],
        foreground=COLORS['text_primary'],
        font=FONTS['body'])
    style.configure('Title.TLabel',
        font=FONTS['title'],
        foreground=COLORS['text_primary'])
    style.configure('Subtitle.TLabel',
        font=FONTS['subtitle'],
        foreground=COLORS['text_primary'])
    style.configure('Secondary.TLabel',
        foreground=COLORS['text_secondary'],
        font=FONTS['small'])
    style.configure('Hint.TLabel',
        foreground=COLORS['text_secondary'],
        font=FONTS['small'])
    style.configure('Success.TLabel',
        foreground=COLORS['success'])
    style.configure('Warning.TLabel',
        foreground=COLORS['warning'])
    style.configure('Error.TLabel',
        foreground=COLORS['danger'])
    
    # ─────────────────────────────────────────────────────────────────
    # Entry (输入框)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TEntry',
        padding=(10, 8),
        fieldbackground=COLORS['bg_primary'],
        foreground=COLORS['text_primary'],
        borderwidth=1,
        relief='solid')
    style.map('TEntry',
        bordercolor=[('focus', COLORS['border_focus']),
                    ('!focus', COLORS['border'])],
        fieldbackground=[('disabled', COLORS['bg_secondary'])])
    
    # ─────────────────────────────────────────────────────────────────
    # Combobox (下拉框)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TCombobox',
        padding=(10, 8),
        fieldbackground=COLORS['bg_primary'],
        foreground=COLORS['text_primary'])
    style.map('TCombobox',
        fieldbackground=[('readonly', COLORS['bg_primary'])])
    
    # ─────────────────────────────────────────────────────────────────
    # Checkbutton (复选框)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TCheckbutton',
        background=COLORS['bg_primary'],
        foreground=COLORS['text_primary'],
        font=FONTS['body'])
    style.map('TCheckbutton',
        background=[('active', COLORS['bg_hover'])])
    
    # ─────────────────────────────────────────────────────────────────
    # Radiobutton (单选框)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TRadiobutton',
        background=COLORS['bg_primary'],
        foreground=COLORS['text_primary'],
        font=FONTS['body'])
    
    # ─────────────────────────────────────────────────────────────────
    # Scale (滑块)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TScale',
        background=COLORS['bg_primary'],
        troughcolor=COLORS['border'],
        sliderlength=20)
    
    # ─────────────────────────────────────────────────────────────────
    # Progressbar (进度条)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TProgressbar',
        background=COLORS['primary'],
        troughcolor=COLORS['border'],
        thickness=6)
    
    # ─────────────────────────────────────────────────────────────────
    # Spinbox (数值框)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TSpinbox',
        padding=(8, 6),
        fieldbackground=COLORS['bg_primary'])
    
    # ─────────────────────────────────────────────────────────────────
    # Separator (分隔线)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TSeparator',
        background=COLORS['border'])
    
    # ─────────────────────────────────────────────────────────────────
    # Scrollbar (滚动条)
    # ─────────────────────────────────────────────────────────────────
    style.configure('TScrollbar',
        background=COLORS['bg_secondary'],
        troughcolor=COLORS['bg_primary'],
        arrowcolor=COLORS['text_secondary'])
    style.map('TScrollbar',
        background=[('active', COLORS['border'])])
    
    return style


def get_button_style(primary=True, danger=False, success=False):
    """获取按钮样式名称"""
    if danger:
        return 'Danger.TButton'
    if success:
        return 'Success.TButton'
    if primary:
        return 'Primary.TButton'
    return 'Secondary.TButton'
