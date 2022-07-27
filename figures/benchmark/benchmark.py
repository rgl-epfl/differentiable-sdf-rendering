"""Figure 9: Benchmark of reverse gradient and redistancing"""
import argparse
import json
import os
import sys
from os.path import join

sys.path.append(os.path.abspath('../'))
from common import *

import tqdm
import configs
from constants import FIGURE_DIR, SCENE_DIR, SDF_DEFAULT_KEY

import mitsuba as mi
import redistancing
from shapes import Grid3d
import time

FIG_NAME = 'benchmark'

def eval_reverse_gradient(scene, config):
    spp = 64
    primal_spp = 4 * spp # same multiplier we use in optimization
    sdf = scene.integrator().sdf
    scene.integrator().warp_field = config.get_warpfield(sdf)
    params = mi.traverse(scene)
    sdf_key = SDF_DEFAULT_KEY
    params.keep([sdf_key])
    param = params[sdf_key]

    dr.enable_grad(param)
    dr.set_grad(param, 0.0)

    # Manually render primal and adjoint image to gather separate timings from kernel history
    dr.kernel_history_clear()
    sensor = 0
    seed = 0
    with dr.suspend_grad():
        img = mi.render(scene, sensor=sensor, seed=seed, spp=primal_spp)

    seed += 1
    dr.enable_grad(img)
    view_loss = dr.sum(img)
    dr.backward(view_loss)
    image_adj = dr.grad(img)
    dr.eval(image_adj)
    dr.disable_grad(img)
    history = dr.kernel_history()
    total_time = sum(h['execution_time'] for h in history)
    stats = {'primal_time': total_time}
    image2 = mi.render(scene, params=params, sensor=sensor, seed=0, spp=primal_spp,
                       seed_grad=seed, spp_grad=config.spp)
    dr.set_grad(image2, image_adj)
    dr.enqueue(dr.ADMode.Backward, image2)
    dr.traverse(mi.Float, dr.ADMode.Backward)
    grad = dr.grad(param)
    dr.eval(grad)
    history = dr.kernel_history()
    total_time = sum(h['execution_time'] for h in history)
    stats['gradient_time'] = total_time
    return img, stats

def sum_stats(total, current, n):
    for k in current.keys():
        if k in total:
            total[k] += current[k] / n
        else:
            total[k] = current[k] / n

def main(force=True):
    dr.set_flag(dr.JitFlag.KernelHistory, True)

    output_dir = join(FIGURE_DIR, FIG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    resx = 256
    resy = 256
    n_runs = 5
    sdfs = ['bunny_128', 'logo_256', 'shadowing_128']
    integrator = 'sdf_direct_reparam'
    sdf_paths = join(SCENE_DIR, 'sdfs')

    # Scene with an envmap
    scene_fn = join(SCENE_DIR, 'bunny', 'bunny.xml')
    used_configs = [configs.OnlyShadingGrad(),
                    configs.Warp(),
                    configs.ConvolutionWarp2(),
                    configs.ConvolutionWarp4(),
                    configs.ConvolutionWarp8(),
                    configs.ConvolutionWarp16(),
                    configs.ConvolutionWarp32()]

    pbar = tqdm.tqdm(total=len(sdfs) * len(used_configs))
    for sdf in sdfs:
        for config in used_configs:
            stats_sum = {}
            stats_fn = join(output_dir, f'{sdf.split("_")[0]}_{config.name}.json')

            if os.path.isfile(stats_fn) and not force:
                print(f'Skipping existing {os.path.split(stats_fn)[-1]}')
                pbar.update()
                continue

            for run in range(n_runs):
                scene = mi.load_file(scene_fn, integrator=integrator, sdf_filename=join(sdf_paths, sdf + '.vol'),
                                 resx=resx, resy=resy, shape_file='dummysdf.xml')
                img, stats = eval_reverse_gradient(scene, config)
                if run == 0:
                    img_fn = join(output_dir, f'{sdf.split("_")[0]}.exr')
                    mi.Bitmap(img).write_async(img_fn)
                del img
                sum_stats(stats_sum, stats, n_runs)

            stats_fn = join(output_dir, f'{sdf.split("_")[0]}_{config.name}.json')
            with open(stats_fn, 'w') as f:
                json.dump(stats_sum, f)
            pbar.update()

    # Additionally, collect performance statistics for fast sweeping method
    resolutions = [16, 32, 64, 128, 256, 512]
    fsm_stats = {}
    for sdf_name in sdfs:
        fsm_stats[sdf_name] = {}
        for res in resolutions:
            # For each resolution, query the SDF and reconstruct it using the
            # FSM redistancing
            r = dr.linspace(mi.Float, 0.0, 1.0, res)
            x, y, z = dr.meshgrid(r, r, r)
            sdf = Grid3d(join(sdf_paths, sdf_name + '.vol'))
            sdf_eval = mi.TensorXf(dr.detach(sdf.eval(mi.Vector3f(y, x, z))), (res, res, res))
            dr.eval(sdf_eval)
            dr.sync_thread()
            total_time = 0.0
            for _ in range(n_runs):
                t0 = time.time()
                sdf = redistancing.redistance(sdf_eval)
                dr.sync_thread()
                total_time += time.time() - t0
            fsm_stats[sdf_name][res] = total_time / n_runs

    stats_fn = join(output_dir, f'fsm.json')
    with open(stats_fn, 'w') as f:
        json.dump(fsm_stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    main(force=args.force)
