"""Figure 15: Comparison of our method to only using shading gradients."""
import argparse
import os
import sys

SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..'))
from common import *

import configs
import tqdm
from constants import OUTPUT_DIR, SCENE_DIR

fig_name = 'shading_gradient'
fig_dir = join(FIGURE_DIR, fig_name)

def main(force=0):
    assert force <= 2
    output_dir = OUTPUT_DIR
    used_configs = [configs.Warp(), configs.OnlyShadingGrad()]

    resx = 1024
    resy = 1024
    spp = 128

    scene_name = 'bunny'
    opt_config = 'no-tex-12-hq'
    translate_y = -0.1
    rotate_y = 260

    # Re-generate the optimization results if needed
    print('[+] Running Optimizations')
    for config in used_configs:
        run_optimization(scene_name, config, opt_config, output_dir, force=force >= 2)

    # For each optimization, render our target view for the visualization in high quality
    print('[+] Rendering views')
    scene_outputs = join(FIGURE_DIR, fig_name, scene_name)
    for config in tqdm.tqdm(used_configs):
        render_optimization_result(scene_name, config, opt_config, image_output_dir=scene_outputs,
                                   resx=resx, resy=resy, spp=spp,
                                   rotate_y=rotate_y, translate_y=translate_y,
                                   output_dir=output_dir, force=force > 0)

    print('[+] Rendering forward mode gradients')
    sdf = 'bunny_128'
    sdf_paths = join(SCENE_DIR, 'sdfs')
    grad_resx = 256
    grad_resy = 256
    techniques = [configs.Warp(), configs.FiniteDifferences(), configs.OnlyShadingGrad()]
    for technique in tqdm.tqdm(techniques):
        scene_fn = join(SCENE_DIR, scene_name, scene_name + '.xml')
        sdf_name = sdf.split("_")[0]
        os.makedirs(join(fig_dir, sdf_name), exist_ok=True)
        img_fn = join(fig_dir, sdf_name, f'{sdf_name}.exr')
        grad_fn = join(fig_dir, sdf_name, f'{sdf_name}_{technique.name}_x.exr')
        if os.path.isfile(img_fn) and os.path.isfile(grad_fn) and force < 1:
            print(f'Skipping existing {os.path.split(grad_fn)[-1]}')
            continue
        scene = mi.load_file(scene_fn, integrator='sdf_direct_reparam', shape_file='dummysdf.xml',
                             sdf_filename=join(sdf_paths, sdf + '.vol'), resx=grad_resx, resy=grad_resy)
        img, grad, _ = eval_forward_gradient(scene, technique, 'x')
        mi.util.write_bitmap(img_fn, img)
        mi.util.write_bitmap(grad_fn, grad)

    mi.Thread.wait_for_tasks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='count', default=0)
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
