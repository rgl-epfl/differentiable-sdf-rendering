"""Figure 11: Compare our method to convolution method"""
import argparse
import os
import sys
from os.path import join

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(join(SCRIPT_DIR, '..'))

from common import *

import configs  # needed to activate integrators
from constants import OUTPUT_DIR
from tqdm.contrib.itertools import product

fig_name = 'conv_opt_results'
fig_dir = join(FIGURE_DIR, fig_name)
used_configs = [configs.Warp(), configs.ConvolutionWarp4(),
                configs.ConvolutionWarp8(), configs.ConvolutionWarp16()]
results = [
    {'scene': 'cubes',
     'opt_config': 'no-tex-6-hqq',
     'y_rotation': -45,
     'y_offset': -0.05,
     'pretty_name': 'Cubes',
     'ref_view': 2,
     'insets': [(0.23, 0.61), (0.41, 0.25)]},

    {'scene': 'chair-diffuse',
     'opt_config': 'diffuse-12-hqq',
     'y_rotation': -130,
     'y_offset': -0.05,
     'pretty_name': 'Chair',
     'ref_view': 4,
     'insets': [(0.05, 0.4), (0.38, 0.42)]},

    {'scene': 'cranium',
     'opt_config': 'no-tex-12-hqq',
     'y_rotation': -60,
     'y_offset': -0.2,
     'pretty_name': 'Cranium',
     'ref_view': 4,
     'insets': [(0.63, 0.65), (0.5, 0.42)]},
]


def main(force=0):
    assert force <= 2
    output_dir = OUTPUT_DIR
    resx = 1024
    resy = 1024
    spp = 256

    # Re-generate the optimization results if needed
    print('[+] Running Optimizations')
    for result in results:
        for config in used_configs:
            run_optimization(result['scene'], config, result['opt_config'],
                             output_dir, force=force >= 2)

    # For each optimization, render our target view for the visualization in high quality
    print('[+] Rendering views')
    for result, config in product(results, used_configs):
        scene_name = result['scene']
        scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
        render_optimization_result(scene_name, config, result['opt_config'], image_output_dir=scene_outputs,
                                   resx=resx, resy=resy, spp=spp,
                                   rotate_y=result['y_rotation'], translate_y=result['y_offset'],
                                   output_dir=output_dir, force=force > 0)
    mi.Thread.wait_for_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
