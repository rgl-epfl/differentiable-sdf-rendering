"""Figure 1: teaser figure"""

import argparse
import os
import sys
from os.path import join

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(join(SCRIPT_DIR, '..'))

from common import *

import configs
import tqdm
from constants import OUTPUT_DIR

fig_name = 'teaser'
fig_dir = join(FIGURE_DIR, fig_name)


def main(force=0):
    assert force <= 2
    output_dir = OUTPUT_DIR

    config = configs.Warp()
    resx = 1024
    resy = 1024
    spp = 256

    # Render all stored iterations
    show_iters = np.arange(0, 512, 64).tolist() + [511, 'final']
    result = {'scene': 'bench',
              'opt_config': 'diffuse-32-hq',
              'y_rotation': -120,
              'y_offset': -0.09}

    # Re-generate the optimization results if needed
    print('[+] Running Optimizations')
    run_optimization(result['scene'], config, result['opt_config'],
                     output_dir, force=force >= 2)

    # For each optimization, render our target view for the visualization in high quality
    print('[+] Rendering views')
    pbar = tqdm.tqdm(len(show_iters) + 2)

    scene_name, opt_config = result['scene'], result['opt_config']
    scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
    kwargs = {'image_output_dir': scene_outputs, 'resx': resx, 'resy': resy, 'spp': spp,
              'rotate_y': result['y_rotation'], 'translate_y': result['y_offset'],
              'output_dir': output_dir, 'force': force > 0}
    render_optimization_result(scene_name, config, opt_config, show_iters, **kwargs, pbar=pbar)

    render_reconstructed_geometry(scene_name, config, opt_config, **kwargs)
    pbar.update()

    render_reference_object(scene_name, opt_config, **kwargs)
    pbar.update()

    mi.Thread.wait_for_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
