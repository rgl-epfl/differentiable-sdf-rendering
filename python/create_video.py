"""This script contains functionality to create a video of the optimization progress of a previously run optimization.
It does not render any new frames and just uses the frames written out during optimization.
"""

import argparse
import glob
import os
import sys
from os.path import join

import mitsuba as mi
import numpy as np
import tqdm

from constants import OUTPUT_DIR
from tempfile import TemporaryDirectory

def gallery(array, ncols=2):
    """Code adapted from https://stackoverflow.com/a/42041135/2351867"""

    n_imgs, height, width, channels = array.shape
    nrows = n_imgs // ncols
    assert n_imgs == nrows * ncols
    return array.reshape(nrows, ncols, height, width, channels) \
        .swapaxes(1, 2) \
        .reshape(height * nrows, width * ncols, channels)

def create_video(output_dir):
    """Create a video from the images in the given directory."""

    from util import run_ffmpeg
    p = os.path.realpath(output_dir)

    images = []
    refs = sorted(glob.glob(join(p, 'ref*.exr')))
    n_sensors = len(refs)

    opt_dir = os.path.join(p, 'opt')
    input_is_ldr = any(fn.endswith('.png') for fn in os.listdir(opt_dir))

    for i in range(n_sensors):
        images.append(sorted(glob.glob(join(opt_dir, f'opt-*-{i:02d}' + ('.png' if input_is_ldr else '.exr')))))

    n_frames = min(len(i) for i in images)
    print(f'[+] Processing {n_frames} opt. frames')
    for j in range(len(images)):
        images[j] = images[j][:n_frames]

    res = mi.Bitmap(refs[0]).size()
    ncols = n_sensors
    nrows = 2
    if ncols / nrows > 4:
        ncols //= 2
        nrows *= 2
    with TemporaryDirectory() as tmp_dir:
        for i in tqdm.tqdm(range(n_frames)):
            frame = np.zeros((2 * n_sensors, res[0], res[1], 3), dtype=np.uint8)
            row = 0
            for s in range(n_sensors):
                row = s // ncols
                img = mi.Bitmap(images[s][i])
                if not input_is_ldr:
                    img = img.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
                frame[2 * row * ncols + s % ncols] = np.array(img)
                ref = mi.Bitmap(refs[s]).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
                frame[(2 * row + 1) * ncols + s % ncols] = np.array(ref)
            frame = gallery(frame, ncols=ncols)
            h = frame.shape[0]
            target_height = 1024
            factor = max(1, target_height // h)
            frame = np.repeat(frame, factor, axis=0)
            frame = np.repeat(frame, factor, axis=1)
            mi.Bitmap(frame).write_async(join(tmp_dir, f'frame-{i:04d}.png'))

        mi.Thread.wait_for_tasks()
        # FFMPEG to create actual video
        frame_name = join(tmp_dir, 'frame-%04d.png')
        video_dir = join(p, 'video')
        os.makedirs(video_dir, exist_ok=True)
        video_path = join(video_dir, 'convergence2.mp4')
        run_ffmpeg(frame_name, video_path)

def main(args):
    mi.set_variant('scalar_rgb')
    parser = argparse.ArgumentParser(description='''Assemble optimization progress video''')
    parser.add_argument('experiments', default=None, nargs='*')
    parser.add_argument('--outputdir', default=OUTPUT_DIR)
    parser.add_argument('--verbose', action='store_true')
    args, uargs = parser.parse_known_args(args)
    for experiment in args.experiments:
        create_video(join(args.outputdir, experiment))

if __name__ == "__main__":
    main(sys.argv[1:])

