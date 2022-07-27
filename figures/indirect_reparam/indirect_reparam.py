"""Figure 13: Optimization results showcasing the benefit of using secondary gradients"""
import argparse
import glob
import os
import subprocess
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))

from common import *

import configs
import tqdm
from constants import OUTPUT_DIR, SCENE_DIR
from opt_configs import get_opt_config
from util import get_checkpoint_path_and_suffix, optimization_result_exists

fig_name = 'indirect_reparam'
fig_dir = join(FIGURE_DIR, fig_name)

results = [
    {'scene': 'torus-shadow',
     'opt_config': 'torus-shadow-1',
     'translate_y': -0.07,
     'rotate_y': -140,
     'configs': [configs.WarpPrimary(), configs.Warp()]},
    {'scene': 'mirror-opt',
     'opt_config': 'mirror-opt-hq',
     'translate_y': -0.07,
     'rotate_y': -140,
     'configs': [configs.WarpPRBPrimary(), configs.WarpPRB()]}
]


def main(force=0, render_turntables=False, verbose=False):
    assert force <= 2
    resx = 1024
    resy = 1024
    spp = 512

    output_dir = OUTPUT_DIR
    cwd = os.path.realpath(join(SCRIPT_DIR, '..', '..', 'python'))

    # Re-generate the optimization results if needed
    print('[+] Running Optimizations')
    for result in results:
        scene, opt_config = result['scene'], result['opt_config']
        for config in result['configs']:
            cfg_name = config.__class__.__name__.lower()
            if optimization_result_exists(output_dir, config, opt_config, scene) and (force < 2):
                print(f"Found result, skipping opt. {scene}/{opt_config}/{cfg_name}/")
                continue
            cmd = f'python optimize.py {scene} --opt {opt_config} --configs {cfg_name} --force'
            # For now run PRB configs using LLVM to deal with long compilation times in Optix
            if 'prb' in config.name:
                cmd += ' --llvm'
            subprocess.call(cmd, shell=True, cwd=cwd)

    # Re-render reference and opt results from different viewpoints at high quality
    render_settings = [0, 1]
    pbar = tqdm.tqdm(range(len(results) * len(results[0]['configs']) * len(render_settings) + 2 * len(results)))
    for result in results:
        scene_name, opt_config = result['scene'], result['opt_config']
        scene_configs = result['configs']
        translate_y, rotate_y = result['translate_y'], result['rotate_y']

        scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
        os.makedirs(scene_outputs, exist_ok=True)
        for config in scene_configs:
            for setting in render_settings:

                scene_config = get_opt_config(opt_config)
                scene_config.scene = scene_name
                p, suffix = get_checkpoint_path_and_suffix(
                    output_dir, scene_name, scene_config.name, config.name)

                fn = join(scene_outputs, f'{config.name}-{setting}.exr')
                if os.path.isfile(fn) and force < 1:
                    print(f"Found output image, skipping rendering. {scene_name}/{opt_config}/{config.name}")
                    pbar.update()
                    continue

                mts_args = {}
                if setting == 0:
                    ref_scene_name = join(SCENE_DIR, scene_name, scene_name + '.xml')
                    shape_file = 'dummysdf.xml'
                else:
                    ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
                    mts_args = {'voffset': translate_y, 'angle': rotate_y}
                    shape_file = 'sdf_shape.xml'

                sdf_filename = sorted(glob.glob(join(p, 'params', '*sdf*.vol')))[-1]
                scene = mi.load_file(ref_scene_name, shape_file=shape_file,
                                     sdf_filename=sdf_filename, integrator=config.integrator,
                                     resx=resx, resy=resy, **mts_args)
                scene_config.load_checkpoint(scene, p, suffix)
                result = mi.render(scene, spp=spp)
                mi.util.write_bitmap(fn, result[..., :4])
                pbar.update()

        # Re-render original reference at high quality
        fn = join(scene_outputs, f'ref.exr')
        if os.path.isfile(fn) and force < 1:
            print(f"Found output image, skipping rendering. {fn}")
            pbar.update()
        else:
            ref_scene_name = join(SCENE_DIR, scene_name, scene_name + '.xml')
            scene = mi.load_file(ref_scene_name, integrator=scene_configs[0].integrator, resx=resx, resy=resy)
            pbar.update()
            result = mi.render(scene, spp=spp)
            mi.util.write_bitmap(fn, result[..., :4])

        # Render the reference geometry (assuming it's contained in [0,1])
        shape_file = scene_name + '-shape.xml'
        fn = join(scene_outputs, f'geo-ref.exr')
        if os.path.isfile(fn) and force < 1:
            print(f"Found output image, skipping rendering. {fn}")
            pbar.update()
            continue
        bsdf_file = 'bsdf_principled.xml' if 'principled' in scene_config.name else 'bsdf_diffuse.xml'
        ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
        scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, bsdf_file=bsdf_file, shape_file=shape_file, sdf_filename='',
                             angle=rotate_y, voffset=translate_y, extra_path=os.path.realpath(join(SCENE_DIR, scene_name)))
        rotate_reference_mesh(scene, translate_y, rotate_y)
        result = mi.render(scene, spp=spp, sensor=0)
        mi.util.write_bitmap(fn, result[..., :4])
        pbar.update()

    # This turntable rendering code is largely duplicated from video.py for now
    # (some of the duplicate code could be extracted to separate functions)
    if render_turntables:
        print('[+] Rendering turntable videos')
        n_frames = 128
        pbar = tqdm.tqdm(range(len(results) * n_frames))
        suffix = 'final'

        for result in results:
            scene_name, opt_config = result['scene'], result['opt_config']
            scene_configs = result['configs']
            translate_y, rotate_y = result['translate_y'], result['rotate_y']
            print(f"scene_name: {scene_name}")
            scene_outputs = join(FIGURE_DIR, fig_name, scene_name, 'turntable')
            os.makedirs(scene_outputs, exist_ok=True)

            for frame in range(n_frames):
                # Compute the momentary rotation of the object
                y_angle = rotate_y + frame / n_frames * 360
                for config in scene_configs:
                    scene_config = get_opt_config(opt_config)
                    scene_config.scene = scene_name
                    config_name = config.name
                    p, _ = get_checkpoint_path_and_suffix(output_dir, scene_name, scene_config.name, config_name)
                    fn = join(scene_outputs, f'{config_name}_{suffix}_{frame:04d}.png')
                    if os.path.isfile(fn) and force < 1:
                        if verbose:
                            print(f"Found output image, skipping rendering. {scene_name}/{opt_config}/{config_name}_{suffix}")
                        continue
                    ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
                    scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, angle=y_angle, voffset=translate_y, fov_axis='y')
                    scene_config.load_checkpoint(scene, p, suffix)
                    result = mi.render(scene, spp=spp)[..., :4]
                    mi.util.write_bitmap(fn, result)

                # Render the reference geometry (assuming it's contained in [0,1])
                shape_file = scene_name + '-shape.xml'
                fn = join(scene_outputs, f'ref_{suffix}_{frame:04d}.png')
                if os.path.isfile(fn) and force < 1:
                    if verbose:
                        print(f"Found output image, skipping rendering. {fn}")
                    pbar.update()
                    continue
                ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
                scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, shape_file=shape_file, sdf_filename='',
                                     angle=y_angle, voffset=translate_y, extra_path=os.path.realpath(join(SCENE_DIR, scene_name)))
                rotate_reference_mesh(scene, translate_y, y_angle)
                result = mi.render(scene, spp=spp)[..., :4]
                mi.util.write_bitmap(fn, result)
                pbar.update()
    mi.Thread.wait_for_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    parser.add_argument('--turntable', action='store_true',
                        help='Renders turntable sequences of the reconstructed geometry')
    parser.add_argument('--verbose', action='store_true', help='Print additional log info')
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force, render_turntables=args.turntable, verbose=args.verbose)
