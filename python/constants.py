"""This file stores the global constants used throughout the code base. Most importantly,
it specifies output paths and scene directory."""

import os

__SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

OUTPUT_DIR = os.path.realpath(os.path.join(__SCRIPT_DIR, '../outputs'))
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

SCENE_DIR = os.path.realpath(os.path.join(__SCRIPT_DIR, '../scenes'))
RENDER_DIR = os.path.realpath(os.path.join(OUTPUT_DIR, 'renders'))

PAPER_DIR = os.path.realpath(os.path.join(__SCRIPT_DIR, '../../diff-sdf-rendering-paper/'))
PAPER_FIG_OUTPUT_DIR = os.path.join(PAPER_DIR, 'figures')

# Default keys for the SDF parameters
SDF_DEFAULT_KEY = 'SamplingIntegrator.sdf.data'
SDF_DEFAULT_KEY_P = 'SamplingIntegrator.sdf.p'

del __SCRIPT_DIR
