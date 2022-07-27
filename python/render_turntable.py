"""This script can be used to render a simple turntable animation of an optimization result."""
import argparse
import sys
from os.path import join

import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

from configs import apply_cmdline_args, get_config
from constants import OUTPUT_DIR, SCENE_DIR
from opt_configs import get_opt_config, is_valid_opt_config
from util import render_turntable, get_checkpoint_path_and_suffix


def create_video(experiment, scene_config, config, output_dir, frames, spp, res, mts_args):
    p, suffix = get_checkpoint_path_and_suffix(output_dir, scene_config.scene,
                                               scene_config.name, config.name)
    scene_name = scene_config.scene
    ref_scene_name = join(SCENE_DIR, scene_name, f'{scene_name}.xml')
    sdf_scene = mi.load_file(ref_scene_name, shape_file='dummysdf.xml',
                             sdf_filename=join(p, 'params', f'sdf-data-{suffix}.vol'),
                             integrator='sdf_direct_reparam', **mts_args)
    scene_config.load_checkpoint(sdf_scene, p, suffix)
    aspect = scene_config.resx / scene_config.resy
    render_turntable(sdf_scene, p, res, int(res / aspect), spp=spp, n_frames=frames)


def main(args):
    parser = argparse.ArgumentParser(description='''Render turntable video''')
    parser.add_argument('scenes', default=None, nargs='*')
    parser.add_argument('--configs', default=['BaseConfig'], type=str, nargs='*')
    parser.add_argument('--optconfigs', '--opt', nargs='+')
    parser.add_argument('--frames', default=64, type=int)
    parser.add_argument('--spp', default=256, type=int)
    parser.add_argument('--res', default=512, type=int)
    parser.add_argument('--outputdir', default=OUTPUT_DIR)
    args, uargs = parser.parse_known_args(args)
    mi.set_log_level(mi.LogLevel.Warn)

    if args.optconfigs is None:
        raise ValueError("Must at least specify one opt. config!")
    assert all(is_valid_opt_config(opt) for opt in args.optconfigs), f"Unknown opt config detected: {args.optconfigs}"

    for scene in args.scenes:
        for opt_config in args.optconfigs:
            for config_name in args.configs:
                config = get_config(config_name)
                remaining_args = apply_cmdline_args(config, uargs, return_dict=True)
                scene_config, mts_args = get_opt_config(opt_config, remaining_args)
                scene_config.scene = scene
                create_video(opt_config, scene_config, config, args.outputdir,
                             args.frames, args.spp, args.res, mts_args)


if __name__ == "__main__":
    main(sys.argv[1:])
