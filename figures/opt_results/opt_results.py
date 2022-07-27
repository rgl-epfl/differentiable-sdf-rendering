"""Figure 10: Main optimization results using our method. Figure shows different
   iterations of the optimization using novel view and lighting conditions."""
import argparse
import os
import sys
from os.path import join
from collections import defaultdict

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(join(SCRIPT_DIR, '..'))

from common import *

import configs  # needed to activate integrators
import tqdm
from constants import OUTPUT_DIR

fig_name = 'opt_results'
fig_dir = join(FIGURE_DIR, fig_name)

results = [
    {'scene': 'dragon',
     'opt_config': 'no-tex-12-hqq',
     'y_rotation': -20,
     'y_offset': -0.2,
     'pretty_name': 'Dragon'},

    {'scene': 'chair',
     'opt_config': 'principled-32-hqq',
     'y_rotation': -130,
     'y_offset': -0.05,
     'pretty_name': 'Chair'},

    {'scene': 'head',
     'opt_config': 'diffuse-16-hqq',
     'y_rotation': -100,
     'y_offset': -0.05,
     'pretty_name': 'Head'},

    {'scene': 'boar',
     'opt_config': 'principled-32-hqq',
     'y_rotation': -130,
     'y_offset': -0.15,
     'pretty_name': 'Boar'},

    {'scene': 'hotdog-diffuse',
     'opt_config': 'diffuse-16-top-hqq',
     'y_rotation': 60,
     'y_offset': -0.2,
     'pretty_name': 'Hotdog'}
]


def main(force=0):
    assert force <= 2
    used_configs = [configs.Warp()]

    spp = 256
    resx = 1024
    resy = 1024
    # Top down sensor for the hotdog scene
    sensor_dict = {'type': 'perspective',
                   'to_world': mi.ScalarTransform4f.look_at(origin=(-0.7, 1.4, 0.0), target=(0.5, 0.1, 0.5), up=(0, 1, 0)),
                   'fov': 39, 'sampler': {'type': 'independent', 'sample_count': 32},
                   'film': {'type': 'hdrfilm', 'width': resx, 'height': resy}}
    sensor_map = defaultdict(lambda: 0)
    sensor_map['hotdog-diffuse'] = mi.load_dict(sensor_dict)
    show_iters = [0, 64, 128, 256, 'final']
    create_and_render_results(results, used_configs, sensor_map, show_iters, spp=spp, resx=resx, resy=resy)


def create_and_render_results(results, used_configs, sensor_map, show_iters,
                              spp, resx=1024, resy=1024, force=0):
    """Runs a set of optimizations and renders the re-lit results"""
    output_dir = OUTPUT_DIR

    # Re-generate the optimization results if needed
    print('[+] Running Optimizations')
    for result in results:
        for config in used_configs:
            run_optimization(result['scene'], config, result['opt_config'],
                             output_dir, force=force >= 2)

    # For each optimization, render our target view for the visualization in high quality
    print('[+] Rendering views')
    pbar = tqdm.tqdm(range(len(results) * len(used_configs) * len(show_iters)))

    for result in tqdm.tqdm(results):
        scene_name, opt_config = result['scene'], result['opt_config']
        sensor = sensor_map[scene_name]
        scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
        for config in used_configs:
            render_optimization_result(scene_name, config, opt_config, show_iters,
                                       image_output_dir=scene_outputs, resx=resx, resy=resy, spp=spp,
                                       rotate_y=result['y_rotation'], translate_y=result['y_offset'],
                                       sensor=sensor, output_dir=output_dir, force=force > 0, pbar=pbar)

        render_reference_object(scene_name, opt_config, image_output_dir=scene_outputs, resx=resx, resy=resy, spp=spp,
                                rotate_y=result['y_rotation'], translate_y=result['y_offset'], sensor=sensor, force=force > 0)
        pbar.update()

    mi.Thread.wait_for_tasks()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
