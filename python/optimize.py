#!/usr/bin/env python3

import argparse, os, sys
from os.path import join

import mitsuba as mi

from constants import OUTPUT_DIR, RENDER_DIR, SCENE_DIR


def render_reference_images(scene_config, config, ref_spp=1024, force=False, verbose=False, mts_args=None):
    """Renders reference images for a given scene (if needed)"""
    from util import set_sensor_res

    scene = scene_config.scene
    ref_scene_name = join(SCENE_DIR, scene, f'{scene}.xml')
    ref_scene = mi.load_file(ref_scene_name, integrator=config.integrator, spp=ref_spp,
                             sdf_filename='', resx=scene_config.resx, resy=scene_config.resy, **mts_args)
    render_folder = join(RENDER_DIR, scene_config.scene, scene_config.name, config.integrator, 'ref')
    os.makedirs(render_folder, exist_ok=True)
    for sensor_idx, sensor in enumerate(scene_config.sensors):
        set_sensor_res(sensor, mi.ScalarVector2i(scene_config.resx, scene_config.resy))
        fn = join(render_folder, f'ref-{sensor_idx:02d}.exr')
        if os.path.isfile(fn) and not force:
            if verbose:
                print(f'File exists, not rendering of {fn}')
        else:
            img = mi.render(ref_scene, sensor=sensor, seed=sensor_idx + 41, spp=ref_spp)
            mi.util.write_bitmap(fn, img[..., :3], write_async=False)

def copy_reference_images_to_output_dir(scene_config, config, output_dir):
    """Copies the reference images to the output directory and returns a list of them"""
    from shutil import copyfile
    ref_image_paths = []
    render_folder = join(RENDER_DIR, scene_config.scene, scene_config.name, config.integrator, 'ref')
    for idx in range(len(scene_config.sensors)):
        ref_name = join(output_dir, f'ref-{idx:02d}.exr')
        copyfile(join(render_folder, f'ref-{idx:02d}.exr'), ref_name)
        ref_image_paths.append(ref_name)
    return ref_image_paths


def optimize(scene_name, config, opt_name, output_dir, ref_spp=1024,
             force=False, verbose=False, opt_config_args=None):
    from opt_configs import get_opt_config
    from shape_opt import optimize_shape

    # 1. Render the reference images if needed
    current_output_dir = join(output_dir, scene_name, opt_name, config.name)
    os.makedirs(current_output_dir, exist_ok=True)
    mi.set_log_level(3 if verbose else mi.LogLevel.Warn)
    opt_config, mts_args = get_opt_config(opt_name, opt_config_args)

    # Pass scene name as part of the opt. config
    opt_config.scene = scene_name
    render_reference_images(opt_config, config, ref_spp=ref_spp, force=force, verbose=verbose, mts_args=mts_args)
    ref_image_paths = copy_reference_images_to_output_dir(opt_config, config, current_output_dir)

    # 2. Optimize SDF compared to ref image(s)
    optimize_shape(opt_config, mts_args, ref_image_paths, current_output_dir, config)


def main(args):
    parser = argparse.ArgumentParser(description='''Reconstructs an object as an SDF''')
    parser.add_argument('scenes', default=None, nargs='*', help='Synthetic reference scenes to optimize')
    parser.add_argument('--optconfigs', '--opt', nargs='+', help='Optimization configurations to run')
    parser.add_argument('--outputdir', default=OUTPUT_DIR, help='Specify the output directory. Default: ../outputs')
    parser.add_argument('--configs', default=['warp'], type=str, nargs='*', help='Method to be used for the optimization. Default: Warp')
    parser.add_argument('--force', action='store_true', help='Force rendering of reference images')
    parser.add_argument('--llvm', action='store_true',
                        help='Force use of LLVM (CPU) mode instead of CUDA/OptiX. This can be useful if compilation times using OptiX are too long.')
    parser.add_argument('--refspp', type=int, default=2048, help='Number of samples per pixel for reference images. Default: 2048')
    parser.add_argument('--verbose', action='store_true', help='Print additional log information')
    parser.add_argument('--print_params', '-pp', action='store_true', help='Print the parameters of the provided scene and exit.')
    args, uargs = parser.parse_known_args(args)

    use_llvm = args.llvm or not ('cuda_ad_rgb' in mi.variants())
    mi.set_variant('llvm_ad_rgb' if use_llvm else 'cuda_ad_rgb')

    from configs import apply_cmdline_args, get_config
    from opt_configs import is_valid_opt_config, get_opt_config

    if args.optconfigs is None:
        raise ValueError('Must at least specify one opt. config!')

    if any(not is_valid_opt_config(opt) for opt in args.optconfigs):
        raise ValueError(f'Unknown opt config detected: {args.optconfigs}')

    for scene_name in args.scenes:
        for config_name in args.configs:
            for opt_config in args.optconfigs:
                config = get_config(config_name)
                remaining_args = apply_cmdline_args(config, uargs, return_dict=True)

                if args.print_params:
                    opt_config, mts_args = get_opt_config(opt_config, remaining_args)
                    print(f'Mitsuba arguments: {mts_args}')
                    sdf_scene = mi.load_file(join(SCENE_DIR, scene_name, f'{scene_name}.xml'),
                                      shape_file='dummysdf.xml', sdf_filename='',
                                      integrator=config.integrator, **mts_args)
                    print('Parameters: ', mi.traverse(sdf_scene))
                    continue

                optimize(scene_name, config, opt_config, args.outputdir, args.refspp, args.force, args.verbose, remaining_args)


if __name__ == '__main__':
    main(sys.argv[1:])
