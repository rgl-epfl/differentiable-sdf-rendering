"""Bundles together some common imports, functions and settings for figure generation / plotting"""

import os
from os.path import join
import subprocess

import sys
__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(join(__SCRIPT_DIR, '../python/'))
sys.path.insert(0, join(__SCRIPT_DIR, '../../../mitsuba3/build/python/'))
# sys.path.insert(0, join(__SCRIPT_DIR, '../../mitsuba3-sdf/build/python/'))
del __SCRIPT_DIR

import numpy as np

import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
mi.set_log_level(mi.LogLevel.Warn)

import drjit as dr
from constants import FIGURE_DIR, PAPER_FIG_OUTPUT_DIR
from result_utils import *

import matplotlib

# Override any style changes by VSCode
if hasattr(matplotlib, "style"):
    matplotlib.style.use('default')

# Use double the true size for figures, as this leads to nicer, thinner plot lines
COLUMN_WIDTH = 2 * 3.36508
TEXT_WIDTH = 2 * 7.06233

tex_fonts = {
    "text.usetex": True,
    "font.size": 12,

    # Slightly smaller font for labels etc
    # "legend.fontsize": 10,
    # "xtick.labelsize": 10,
    # "ytick.labelsize": 10,
    "text.latex.preamble": r"""\usepackage{libertine}
                               \usepackage{amsmath}
                               \usepackage{bm}""",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

matplotlib.rcParams.update(tex_fonts)

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def read_img(fn, exposure=0, tonemap=True, background_color=None,
             handle_inexistant_file=False):
    """Reads an image from a file and (optionally) tonemaps it or composites it
    over a constant colored background."""
    if handle_inexistant_file and not os.path.isfile(fn):
        return np.ones((256, 256, 3)) * 0.3
    bmp = mi.Bitmap(fn)
    if tonemap:
        if background_color is not None:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.Float32, False))
            background_color = np.array(background_color).ravel()[None, None, :]
            # img = img[:, :, :3] * img[..., -1][..., None] + (1.0 - img[..., -1][..., None]) * background_color
            img = img[:, :, :3] + (1.0 - img[..., -1][..., None]) * background_color
        else:
            img = np.array(bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False))
        img = img * 2 ** exposure

        return np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)), 0, 1)
    else:
        return np.array(bmp)


def tonemap(img):
    """Tonemaps an image to sRGB color space"""
    return np.clip(np.array(mi.Bitmap(img).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, True)), 0, 1)


def save_fig(fig_name, dpi=300, pad_inches=0.005, bbox_inches='tight', compress=True, file_format='pdf', **kwargs):
    """Saves a figure as a PDF. By default, the figure will be compressed using
    ghost script to reduce the final file size and PDF load times."""
    import shutil
    output_dir = os.path.join(PAPER_FIG_OUTPUT_DIR, fig_name)
    os.makedirs(output_dir, exist_ok=True)
    fn = join(output_dir, fig_name + '.' + file_format)

    compress = compress and (file_format == 'pdf') # only compress PDF files
    if compress and (shutil.which('gs') is None):
        print("Cannot find ghost script ('gs'), skipping compression of PDF output")
        compress = False

    if compress:
        fn = fn.replace('.pdf', '_uc.pdf')
    plt.savefig(fn, format=file_format, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, **kwargs)

    if compress:
        gs = f"gs -o {fn.replace('_uc.pdf', '.pdf')} -dQUIET -f -dNOPAUSE -dBATCH "
        gs += "-sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dCompatibilityLevel=1.6 "
        gs += f"-dDownsampleColorImages=false -DownsampleGrayImages=false {fn}"
        subprocess.call(gs, shell=True)


def disable_ticks(ax):
    """Disable ticks around plot (useful for displaying images)"""
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])


def disable_border(ax):
    """Disable border around plot"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def apply_color_map(data, cmap='coolwarm', vmin=None, vmax=None):
    """Applies a matplotlib color map to an array. Optionally allows to specify vmin and vmax."""
    from matplotlib import cm
    data = np.array(data)
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.min(data)
    return getattr(cm, cmap)(plt.Normalize(vmin, vmax)(data))[..., :3]


def time_to_string(duration):
    """Converts a duration in seconds to a human-readable string."""
    duration = round(duration)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    result = ''
    if d > 0:
        result += f'{d}d '
    if h > 0:
        result += f'{h}h '
    if m > 0:
        result += f'{m}m '
    result += f'{s}s'
    return result


def smooth_loss(values, smoothing_weight=0.5, debias=True):
    """Computes exponential moving average with de-biasing following the implementation
       in TensorBoard (https://github.com/tensorflow/tensorboard)"""
    prev = 0.0
    smoothed = []
    for i, value in enumerate(values):
        prev = prev * smoothing_weight + (1 - smoothing_weight) * value
        smoothed.append(prev / (1 - smoothing_weight ** (i + 1)) if debias else prev)
    return np.array(smoothed)
