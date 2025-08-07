#!/usr/bin/env python3
"""
Modern figure style configuration for PhD dissertation.
Provides minimal, professional aesthetic for all research figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def set_dissertation_style():
    """
    Apply modern, minimal style settings for dissertation figures.
    Suitable for PhD-level academic publications.
    """
    
    # Use seaborn whitegrid as base (cleaner than default matplotlib)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Custom style parameters for modern, minimal look
    style_params = {
        # Font settings - professional sans-serif
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        
        # Figure settings
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'figure.figsize': (10, 6),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'none',
        
        # Axes settings - minimal borders
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#CCCCCC',
        'axes.linewidth': 0.8,
        'axes.labelcolor': '#333333',
        'axes.axisbelow': True,
        'axes.facecolor': 'white',
        
        # Grid settings - subtle
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,
        'grid.color': '#E0E0E0',
        
        # Tick settings
        'xtick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.major.size': 4,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.color': '#666666',
        'ytick.color': '#666666',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#CCCCCC',
        'legend.borderpad': 0.5,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 1.5,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.5,
        'patch.linewidth': 0.5,
        
        # Error bar settings
        'errorbar.capsize': 3,
        
        # Histogram settings
        'hist.bins': 'auto',
        
        # Image settings
        'image.cmap': 'viridis',
    }
    
    plt.rcParams.update(style_params)
    
    # Set default color cycle to a more muted palette
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        '#4A90E2',  # Soft blue
        '#F5A623',  # Muted orange
        '#7ED321',  # Soft green
        '#9B9B9B',  # Gray
        '#D0021B',  # Soft red
        '#9013FE',  # Purple
        '#50E3C2',  # Cyan
        '#B8B8B8',  # Light gray
    ])

# Modern, professional color palette for dissertation
COLORS = {
    'baseline': '#4A90E2',      # Soft blue - primary color
    'synthetic': '#F5A623',     # Muted orange - secondary color  
    'hybrid': '#7ED321',        # Soft green - tertiary color
    'neutral': '#9B9B9B',       # Professional gray
    'highlight': '#D0021B',     # Soft red for emphasis
    'light_gray': '#F0F0F0',    # Background gray
    'dark_gray': '#4A4A4A',     # Text gray
    'accent': '#9013FE',        # Purple accent
    'success': '#7ED321',       # Green for positive
    'warning': '#F8E71C',       # Yellow for warning
    'error': '#D0021B',         # Red for negative
    'info': '#4A90E2',          # Blue for information
}

# Additional color variations for complex plots
COLORS_ALPHA = {
    'baseline_light': '#4A90E233',      # 20% opacity
    'synthetic_light': '#F5A62333',     # 20% opacity
    'hybrid_light': '#7ED32133',        # 20% opacity
    'baseline_medium': '#4A90E266',     # 40% opacity
    'synthetic_medium': '#F5A62366',    # 40% opacity
    'hybrid_medium': '#7ED32166',       # 40% opacity
}

def get_color(name, alpha=None):
    """
    Get a color from the palette with optional transparency.
    
    Args:
        name: Color name from COLORS dictionary
        alpha: Optional alpha value (0-1) for transparency
    
    Returns:
        Color hex code with optional alpha
    """
    color = COLORS.get(name, '#000000')
    if alpha is not None:
        # Convert hex to RGBA
        from matplotlib.colors import to_rgba
        rgba = to_rgba(color)
        return (*rgba[:3], alpha)
    return color

def apply_minimal_style(ax):
    """
    Apply additional minimal styling to an axis.
    
    Args:
        ax: Matplotlib axis object
    """
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Make remaining spines subtle
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    
    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean tick marks
    ax.tick_params(length=4, width=0.8, colors='#666666')
    
    return ax

def format_axis_labels(ax, xlabel=None, ylabel=None, title=None):
    """
    Format axis labels with consistent styling.
    
    Args:
        ax: Matplotlib axis object
        xlabel: X-axis label text
        ylabel: Y-axis label text
        title: Plot title text
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color='#333333', fontweight='normal')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color='#333333', fontweight='normal')
    if title:
        ax.set_title(title, fontsize=12, color='#333333', fontweight='medium', pad=15)
    
    return ax

def add_value_labels(ax, bars, format_str='{:.3f}', offset=0.01, fontsize=8):
    """
    Add value labels to bar chart with consistent formatting.
    
    Args:
        ax: Matplotlib axis object
        bars: Bar container from ax.bar()
        format_str: Format string for values
        offset: Vertical offset for labels
        fontsize: Font size for labels
    """
    for bar in bars:
        height = bar.get_height()
        if height != 0:  # Only label non-zero bars
            ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                   format_str.format(height),
                   ha='center', va='bottom', fontsize=fontsize, color='#666666')
    
    return ax

# Export all style components
__all__ = [
    'set_dissertation_style',
    'COLORS',
    'COLORS_ALPHA',
    'get_color',
    'apply_minimal_style',
    'format_axis_labels',
    'add_value_labels'
]