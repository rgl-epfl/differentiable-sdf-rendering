"""Runs all the (potentially long-running) figure generation scripts.

The Jupyter notebooks to generate the final figure PDFs still have to be run manually.
"""

import os
from os.path import join
import subprocess

fig_names = ['benchmark', 'forwardgrads', 'nested_reparam',
             'normalization', 'sdf',  'sphere_reparam', 'opt_results',
             'shading_gradients', 'teaser', 'conv_opt_results', 'indirect_reparam', 'limitations']

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))

for fig in fig_names:
    fig_folder = join(SCRIPT_DIR, fig)
    py_file = join(fig_folder, fig + '.py')
    if not os.path.isfile(py_file):
        print(f"No file {fig}.py found, skipping")
        continue
    print("Running:", py_file)
    cmd = f'python {fig}.py'
    subprocess.call(cmd, shell=True, cwd=fig_folder)
