"""Figure 4: Effect of normalizing warp field by squared norm"""
import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from common import *

import mesh_to_sdf
import shapes
import configs

from constants import SCENE_DIR

import tqdm


def render_normalization_figure(force=0):
    """Compute gradients using both normalized and unnormalized warp fields,
       and at several SDF resolutions."""

    figname = 'normalization'
    output_dir = join(FIGURE_DIR, figname)
    os.makedirs(output_dir, exist_ok=True)
    resx = 720
    resy = 720
    fd_spp = 1024
    mesh_fn = join(SCENE_DIR, 'dragon', 'meshes', 'dragon.ply')
    sdf_resolutions = [64, 128, 256]
    used_configs = [configs.Warp(), configs.WarpNotNormalized(), configs.FiniteDifferences()]
    ref_scene_name = join(SCENE_DIR, 'figures', 'normalization', 'normalization.xml')
    pbar = tqdm.tqdm(range(len(sdf_resolutions) * len(used_configs)))
    for sdf_res in sdf_resolutions:
        sdf = shapes.Grid3d(mesh_to_sdf.create_sdf(mesh_fn, sdf_res))
        for config in used_configs:
            grad_fn = join(output_dir, f'{config.name}_{sdf_res}_grad.exr')
            if os.path.isfile(grad_fn) and not force > 0:
                print("Skipping, file exists: ", f'{config.name}_{sdf_res}_grad.exr')
                pbar.update()
                continue

            scene = mi.load_file(ref_scene_name, resx=resx, resy=resy)
            scene.integrator().sdf = sdf
            img, grad, _ = eval_forward_gradient(scene, config, 'y', fd_spp=fd_spp)
            mi.util.write_bitmap(grad_fn, grad)
            mi.util.write_bitmap(join(output_dir, f'{config.name}.exr'), img)
            pbar.update()

    mi.Thread.wait_for_tasks()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    render_normalization_figure(force=args.force)
