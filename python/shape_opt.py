import time

import os
from os.path import join

import drjit as dr
import mitsuba as mi
import numpy as np
import tqdm

from constants import SCENE_DIR
from create_video import create_video
from util import dump_metadata, render_turntable, resize_img, set_sensor_res


def load_ref_images(paths, multiscale=False):
    """Load the reference images and compute scale pyramid for multiscale loss"""
    if not multiscale:
        return [mi.TensorXf(mi.Bitmap(fn)) for fn in paths]
    result = []
    for fn in paths:
        bmp = mi.Bitmap(fn)
        d = {int(bmp.size()[0]): mi.TensorXf(bmp)}
        new_res = bmp.size()
        while np.min(new_res) > 4:
            new_res = new_res // 2
            d[int(new_res[0])] = mi.TensorXf(resize_img(bmp, new_res, smooth=True))
        result.append(d)
    return result


def optimize_shape(scene_config, mts_args, ref_image_paths,
                   output_dir, config, write_ldr_images=True):
    """Main function that runs the actual SDF shape reconstruction"""

    if len(mts_args) > 0:
        print(f"Cmdline arguments passed to Mitsuba: {mts_args}")

    scene_name = scene_config.scene
    ref_scene_name = join(SCENE_DIR, scene_name, f'{scene_name}.xml')
    ref_images = load_ref_images(ref_image_paths, True)

    # Load scene, currently handle SDF shape separately from Mitsuba scene
    sdf_scene = mi.load_file(ref_scene_name, shape_file='dummysdf.xml', sdf_filename=join(SCENE_DIR, 'sdfs', 'bunny_64.vol'),
                             integrator=config.integrator, resx=scene_config.resx, resy=scene_config.resy, **mts_args, parallel=config.use_parallel_loading)
    sdf_object = sdf_scene.integrator().sdf
    sdf_scene.integrator().warp_field = config.get_warpfield(sdf_object)

    params = mi.traverse(sdf_scene)
    assert any('_sdf_' in shape.id() for shape in sdf_scene.shapes()), "Could not find a placeholder shape for the SDF"
    params.keep(scene_config.param_keys)

    opt = mi.ad.Adam(lr=config.learning_rate, params=params, mask_updates=config.mask_optimizer)
    n_iter = config.n_iter
    scene_config.initialize(opt, sdf_scene)
    params.update(opt)

    # Render shape initialization
    for idx, sensor in enumerate(scene_config.sensors):
        with dr.suspend_grad():
            img = mi.render(sdf_scene, sensor=sensor, seed=idx,
                            spp=config.spp * config.primal_spp_mult)
        mi.util.write_bitmap(join(output_dir, f'init-{idx:02d}.exr'), img[..., :3])

    # Set initial rendering resolution
    for sensor in scene_config.sensors:
        set_sensor_res(sensor, scene_config.init_res)

    opt_image_dir = join(output_dir, 'opt')
    os.makedirs(opt_image_dir, exist_ok=True)
    seed = 0
    loss_values = []
    try:
        pbar = tqdm.tqdm(range(n_iter))
        for i in pbar:
            loss = mi.Float(0.0)
            for idx, sensor in scene_config.get_sensor_iterator(i):
                img = mi.render(sdf_scene, params=params, sensor=sensor,
                                seed=seed, spp=config.spp * config.primal_spp_mult,
                                seed_grad=seed + 1 + len(scene_config.sensors), spp_grad=config.spp)
                seed += 1 + len(scene_config.sensors)
                view_loss = scene_config.loss(img, ref_images[idx][sensor.film().crop_size()[0]]) / scene_config.batch_size
                dr.backward(view_loss)
                bmp = resize_img(mi.Bitmap(img), scene_config.target_res)
                mi.util.write_bitmap(join(opt_image_dir, f'opt-{i:04d}-{idx:02d}' + ('.png' if write_ldr_images else '.exr')), bmp)
                loss += view_loss

            # Evaluate regularization loss
            reg_loss = scene_config.eval_regularizer(opt, sdf_object, i)
            if dr.grad_enabled(reg_loss):
                dr.backward(reg_loss)
            loss += dr.detach(reg_loss)

            scene_config.save_params(opt, output_dir, i, force=i == n_iter - 1)
            scene_config.validate_gradients(opt, i)
            loss_str = f'Loss: {loss[0]:.4f}'
            if dr.grad_enabled(reg_loss):
                loss_str += f' (reg (avg. x 1e4): {1e4*reg_loss[0] / dr.prod(sdf_object.shape):.4f})'

            pbar.set_description(loss_str)
            loss_values.append(loss[0])
            opt.step()
            scene_config.validate_params(opt, i)
            scene_config.update_scene(sdf_scene, i)
            params.update(opt)
    finally:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(loss_values)), loss_values)
        plt.xlabel('Iterations')
        plt.ylabel('Objective function value')
        avg_loss = np.mean(np.array(loss_values)[-5:])
        plt.title(f"Final loss: {100*loss_values[-1]:.3f} (avg. over 5 its: {100*avg_loss:.3f})")
        plt.savefig(join(output_dir, 'loss.pdf'))
        plt.savefig(join(output_dir, 'loss.png'))

        # Write out total time and basic config info to json
        d = {'total_time': time.time() - pbar.start_t, 'loss_values': loss_values}
        dump_metadata(config, scene_config, d, join(output_dir, 'metadata.json'))

    # If optimization finished, create optimization video and turntable anim
    print("[+] Writing convergence video")
    create_video(output_dir)
    print("[+] Rendering turntable")
    # Load the exponential moving average of the parameters and save them, and render turntable
    if scene_config.param_averaging_beta is not None:
        scene_config.load_mean_parameters(opt)
        scene_config.save_params(opt, output_dir, 'final')
        params.update(opt)

    sdf_scene.integrator().warp_field = None
    render_turntable(sdf_scene, output_dir, resx=512, resy=512, spp=256, n_frames=64)
