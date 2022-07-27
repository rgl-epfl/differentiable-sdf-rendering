"""Figure 16: Limitations when reconstructing complex geometry (Lego excavator)"""
import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from common import *

import configs  # needed to activate integrators
import tqdm
from constants import OUTPUT_DIR, SCENE_DIR
from opt_configs import get_opt_config
from util import get_checkpoint_path_and_suffix, get_regular_cameras

fig_name = 'limitations'
fig_dir = join(FIGURE_DIR, fig_name)

def main(force=0):
    assert force <= 2
    resx = 1024
    resy = 1024
    spp = 512
    output_dir = OUTPUT_DIR
    scene_name = 'lego'
    opt_config = 'diffuse-40-hqq'
    config = configs.Warp()
    sensor = get_regular_cameras(16, resx=resx, resy=resy)[0]

    print('[+] Running Optimizations')
    run_optimization(scene_name, config, opt_config, output_dir, force=force >= 2)

    scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
    os.makedirs(scene_outputs, exist_ok=True)
    scene_config = get_opt_config(opt_config)
    scene_config.scene = scene_name
    p, suffix = get_checkpoint_path_and_suffix(output_dir, scene_name,
                                               scene_config.name, config.name)
    print('[+] Rendering images')
    fn = join(scene_outputs, f'{config.name}.exr')
    if os.path.isfile(fn) and force < 1:
        print(f"Found output image, skipping rendering. {fn}")
    else:
        mts_args = {}
        ref_scene_name = join(SCENE_DIR, scene_name, scene_name + '.xml')
        sdf_filename = sorted(glob.glob(join(p, 'params', '*sdf*.vol')))[-1]
        scene = mi.load_file(ref_scene_name, shape_file='dummysdf.xml', sdf_filename=sdf_filename,
                             integrator=config.integrator, resx=resx, resy=resy, **mts_args)
        scene_config.load_checkpoint(scene, p, suffix)
        result = mi.render(scene, spp=spp, sensor=sensor)
        mi.util.write_bitmap(fn, result[..., :4])

    # Re-render original reference at high quality
    fn = join(scene_outputs, f'ref.exr')
    if os.path.isfile(fn) and force < 1:
        print(f"Found output image, skipping rendering. {fn}")
    else:
        ref_scene_name = join(SCENE_DIR, scene_name, scene_name + '.xml')
        scene = mi.load_file(ref_scene_name, integrator=config.integrator, resx=resx, resy=resy)
        result = mi.render(scene, spp=spp, sensor=sensor)
        mi.util.write_bitmap(fn, result[..., :4])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
