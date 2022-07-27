import os
import subprocess

from os.path import join

import drjit as dr
import mitsuba as mi

from constants import OUTPUT_DIR, SCENE_DIR, SDF_DEFAULT_KEY_P
from util import get_checkpoint_path_and_suffix, optimization_result_exists
from opt_configs import get_opt_config

def rotate_reference_mesh(scene, translate_y, y_angle):
    params = mi.traverse(scene)
    valid_keys = [k for k in params.keys() if k.endswith('.vertex_positions') and not k.startswith('studio-')]
    params.keep(valid_keys)
    tf = mi.ScalarTransform4f.translate([0.5, 0.5 + translate_y, 0.5]) \
                             .rotate([0, 1, 0], y_angle) \
                             .translate([-0.5, -0.5, -0.5])
    for k in valid_keys:
        params[k] = dr.ravel(mi.Transform4f(tf) @ dr.unravel(mi.Point3f, params[k]))
    params.update()


def run_optimization(scene, config, opt_config, output_dir=OUTPUT_DIR, force=False, dry_run=False):
    SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
    cwd = os.path.realpath(os.path.join(SCRIPT_DIR, '..', 'python'))

    cfg_name = config.__class__.__name__.lower()
    if optimization_result_exists(output_dir, config, opt_config, scene) and not force:
        print(f"Found result, skipping opt. {scene}/{opt_config}/{cfg_name}/")
        return

    cmd = f'python optimize.py {scene} --opt {opt_config} --configs {cfg_name} --outputdir {output_dir} --force'
    if not dry_run:
        subprocess.call(cmd, shell=True, cwd=cwd)
    else:
        print(cmd)


def render_optimization_result(scene_name, config, opt_config, show_iters=None,
                               image_output_dir=None,
                               resx=1024, resy=1024, spp=256,
                               rotate_y=0, translate_y=0,
                               sensor=0, output_dir=OUTPUT_DIR, force=False, pbar=None,
                               image_fn=None):
    """Renders the optimization result using a studio lighting setup"""
    ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')

    if isinstance(opt_config, str):
        opt_config = get_opt_config(opt_config)

    os.makedirs(image_output_dir, exist_ok=True)

    bsdf_file = 'bsdf_principled.xml' if 'principled' in opt_config.name else 'bsdf_diffuse.xml'
    opt_config.scene = scene_name
    config_name = config.name

    p, suffix = get_checkpoint_path_and_suffix(output_dir, scene_name, opt_config.name, config_name)
    if show_iters is None:
        show_iters = [suffix]
    for show_iter in show_iters:
        suffix = f'{show_iter:04d}' if isinstance(show_iter, int) else show_iter
        if image_fn is not None:
            fn = image_fn
        else:
            fn = join(image_output_dir, f'{config_name}_{suffix}.exr')
        if os.path.isfile(fn) and not force:
            print(f"Found output image, skipping rendering. {scene_name}/{opt_config.name}/{config_name}_{suffix}")
            if pbar is not None:
                pbar.update()
            continue
        scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, bsdf_file=bsdf_file,
                             angle=rotate_y, voffset=translate_y)
        opt_config.load_checkpoint(scene, p, suffix)
        result = mi.render(scene, spp=spp, sensor=sensor)
        mi.util.write_bitmap(fn, result)
        if pbar is not None:
            pbar.update()


def render_reconstructed_geometry(scene_name, config, opt_config, image_output_dir, output_dir=OUTPUT_DIR,
                                  resx=1024, resy=1024, spp=256,
                                  rotate_y=0, translate_y=0, sensor=0, force=False):
    ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
    if isinstance(opt_config, str):
        opt_config = get_opt_config(opt_config)

    p, suffix = get_checkpoint_path_and_suffix(output_dir, scene_name,
                                               opt_config.name, config.name)
    fn = join(image_output_dir, f'{config.name}_final_geo.exr')
    if os.path.isfile(fn) and not force:
        print(f"Found output image, skipping rendering. {scene_name}/{opt_config.name}/{config.name}_final_geo")
        return
    scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, bsdf_file='bsdf_plain.xml',
                         angle=rotate_y, voffset=translate_y)
    opt_config.load_checkpoint(scene, p, suffix)
    result = mi.render(scene, spp=spp, sensor=sensor)
    mi.util.write_bitmap(fn, result)


def render_reference_object(scene_name, opt_config, image_output_dir,
                              resx=1024, resy=1024, spp=256,
                              rotate_y=0, translate_y=0, sensor=0, force=False, **kwargs):
    ref_scene_name = join(SCENE_DIR, 'figures', 'studio', 'studio.xml')
    os.makedirs(image_output_dir, exist_ok=True)
    if isinstance(opt_config, str):
        opt_config = get_opt_config(opt_config)

    shape_file = scene_name + '-shape.xml'
    extra_path = os.path.realpath(join(SCENE_DIR, scene_name))
    fn = join(image_output_dir, f'reference.exr')
    if os.path.isfile(fn) and not force:
        print(f"Found output image, skipping rendering. {scene_name}/{opt_config.name}/reference")
        return
    bsdf_file = 'bsdf_principled.xml' if 'principled' in opt_config.name else 'bsdf_diffuse.xml'

    scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, bsdf_file=bsdf_file,
                         shape_file=shape_file, sdf_filename='', angle=rotate_y,
                         voffset=translate_y, extra_path=extra_path)
    rotate_reference_mesh(scene, translate_y, rotate_y)
    result = mi.render(scene, spp=spp, sensor=sensor)
    mi.util.write_bitmap(fn, result)


def eval_forward_gradient(scene, config, axis='x', spp=1024, fd_spp=8192, fd_eps=1e-3, sensor=0):
    """Evalutes a forward gradient image for a given axis"""
    sdf = scene.integrator().sdf
    params = mi.traverse(scene)
    params.keep([SDF_DEFAULT_KEY_P])

    if axis == 'x':
        param = params[SDF_DEFAULT_KEY_P].x
    elif axis == 'y':
        param = params[SDF_DEFAULT_KEY_P].y
    else:
        param = params[SDF_DEFAULT_KEY_P].z

    dr.enable_grad(param)
    dr.set_grad(param, 0.0)
    dr.eval(params[SDF_DEFAULT_KEY_P])
    dr.kernel_history_clear()
    if config.use_finite_differences:
        with dr.suspend_grad():
            img = mi.render(scene, integrator=scene.integrator(), sensor=sensor, spp=fd_spp)
            param += fd_eps
            img2 = mi.render(scene, integrator=scene.integrator(), sensor=sensor, spp=fd_spp)
            grad = (img2 - img) / fd_eps
            dr.eval(grad)
    else:
        scene.integrator().warp_field = config.get_warpfield(sdf)
        img = mi.render(scene, params=params, sensor=sensor,
                        integrator=scene.integrator(), spp=spp)
        dr.forward(param)
        grad = dr.grad(img)
        dr.eval(grad)

    history = dr.kernel_history()
    total_time = sum(h['execution_time'] for h in history)
    stats = {'total_time': total_time}
    return img, grad, stats