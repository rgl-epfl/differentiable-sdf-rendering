"""Figure 8: Evaluate gradients in forward mode using several different methods"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from common import *

import tqdm
import configs
from constants import FIGURE_DIR, SCENE_DIR

FIG_NAME = 'forwardgrads'


def main(force=True):
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    resx = 128
    resy = 128
    scenes = ['glossy_plane', 'plane_red_object', 'plane_area']
    sdfs = ['bunny_128', 'logo_256', 'shadowing_128']
    emitters = ['', 'emitters/cathedral.xml', 'emitters/vmf.xml']
    integrators = ['sdf_prb_reparam', 'sdf_direct_reparam', 'sdf_direct_reparam']
    params = ['x', 'x', 'x']
    sdf_paths = join(SCENE_DIR, 'sdfs')
    techniques = [configs.Warp(), configs.FiniteDifferences(), configs.ConvolutionWarp2(),
                  configs.ConvolutionWarp4(), configs.ConvolutionWarp8(),
                  configs.ConvolutionWarp16(), configs.ConvolutionWarp32()]
    output_dir = join(FIGURE_DIR, FIG_NAME)
    os.makedirs(output_dir, exist_ok=True)
    pbar = tqdm.tqdm(total=len(scenes) * len(techniques))
    for axis, scene_name, sdf, emitter, integrator in tqdm.tqdm(zip(params, scenes, sdfs, emitters, integrators)):
        for technique in techniques:
            scene_fn = join(SCENE_DIR, scene_name, scene_name + '.xml')
            img_fn = join(output_dir, f'{sdf.split("_")[0]}.exr')
            grad_fn = join(output_dir, f'{sdf.split("_")[0]}_{technique.name}_{axis}.exr')
            if os.path.isfile(img_fn) and os.path.isfile(grad_fn) and not force:
                print(f'Skipping existing {os.path.split(grad_fn)[-1]}')
                pbar.update()
                continue

            scene = mi.load_file(scene_fn, integrator=integrator, sdf_filename=join(sdf_paths, sdf + '.vol'),
                                 resx=resx, resy=resy, emitter_scene=emitter)
            img, grad, stats = eval_forward_gradient(scene, technique, axis)
            mi.util.write_bitmap(img_fn, img)
            mi.util.write_bitmap(grad_fn, grad)
            stats_fn = join(output_dir, f'{sdf.split("_")[0]}_{technique.name}_{axis}.json')
            with open(stats_fn, 'w') as f:
                json.dump(stats, f)
            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
