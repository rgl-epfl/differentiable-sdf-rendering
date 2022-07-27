"""Figure 3: SDF level set visualization"""
import sys
import os
from os.path import join
sys.path.append(os.path.abspath('../../python/'))

import sys
sys.path.append(os.path.abspath('../'))
from common import *

import mitsuba as mi
import mesh_to_sdf
import shapes
import configs # needed to activate integrators
from matplotlib import cm

from constants import SCENE_DIR, FIGURE_DIR


def render_sdf_isolines_figure():
    fig_name = 'sdf'
    output_dir = join(FIGURE_DIR, fig_name)
    os.makedirs(output_dir, exist_ok=True)

    use_bunny_scene = False

    if use_bunny_scene:
        use_file_sensor = True
        mesh_fn = join(SCENE_DIR, 'meshes', 'bunny2.obj')
        bsdf_name = 'grey_bsdf'
        y_scale = 0.5
        resx = 1024
        resy = 1024
        ref_scene_name = join(SCENE_DIR, 'figures', 'sdf', 'sdf_bunny.xml')
        rotate_y = -15
        spp = 512
    else:
        use_file_sensor = False
        mesh_fn = join(SCENE_DIR, 'extra', 'meshes', 'xyzrgb_dragon.ply')
        bsdf_name = 'plastic_bsdf'
        y_scale = 0.3
        resx = 1280
        resy = 720
        ref_scene_name = join(SCENE_DIR, 'figures', 'sdf', 'sdf.xml')
        rotate_y = -20
        spp = 1024

    mesh_type = os.path.splitext(mesh_fn)[1][1:]

    print("[+] Creating SDF...")
    sdf = mesh_to_sdf.create_sdf(mesh_fn, 256)
    sdf_shape = shapes.Grid3d(sdf)

    if use_bunny_scene:
        sdf_shape.p = mi.Vector3f(-0.5, -0.5, -0.5)

    # Render SDF with illumination and transparent background
    res_scale = 1.0
    resx = int(resx * res_scale)
    resy = int(resy * res_scale)

    r_min = -1.0
    r_max = 1.0
    vmax = 0.2
    vmin = -vmax
    size = r_max - r_min
    sensor = mi.load_dict({
        'type': 'perspective',
                'fov': 38.0, 'sampler': {'type': 'independent'},
                'film': {'type': 'hdrfilm', 'width': resx, 'height': resy,
                         'pixel_filter': {'type': 'gaussian'}, 'pixel_format': 'rgba'},
                'to_world': mi.ScalarTransform4f.look_at([-0.35, 1, 1.8], [0.45, 0.5, 0.5], [0, 1, 0])})

    if use_file_sensor:
        sensor = 0
    sdf_scene = mi.load_file(ref_scene_name, resx=resx, resy=resy, mesh_path=mesh_fn, mesh_type=mesh_type,
                              integrator_file="integrator_path.xml", spp=spp, bsdf_name=bsdf_name,
                              rect_scale=size / 2, max_depth=8, rotate_y=rotate_y)
    params = mi.traverse(sdf_scene)
    tex_key = "Rectangle.bsdf.reflectance.data"
    rect_trafo_key = "Rectangle.to_world"

    # Evaluate the SDF on the plane
    res = 1024
    x, y = dr.meshgrid(dr.linspace(mi.Float, r_min, r_max, res),
                       dr.linspace(mi.Float, r_min, r_max, res), indexing='ij')
    eval_p = mi.Point3f(y, x, 0.0)

    if use_bunny_scene:
        rect_trafo = mi.ScalarTransform4f.translate(mi.ScalarVector3f(0.0, 0.0, -0.05)).rotate([0, 1, 0], rotate_y).scale([0.5, y_scale, 0.5])
    else:
        rect_trafo = mi.ScalarTransform4f.translate(mi.ScalarVector3f(0.51, 0.47, 0.45)).rotate([0, 1, 0], rotate_y).scale([0.5, y_scale, 0.5])
    values = sdf_shape.eval(mi.Transform4f(rect_trafo) @ eval_p)
    params[rect_trafo_key] = rect_trafo
    isoline_spacing = 0.035
    isoline_thickness = 0.1

    # Map SDF values to an actual color
    if True:
        values_rgb = mi.Vector3f(cm.coolwarm(plt.Normalize(vmin, vmax)(np.array(values)))[..., :3] ** 2.2)
        values_rgb = dr.select((dr.abs(values) / isoline_spacing) - dr.floor(dr.abs(values) /
                               isoline_spacing) < isoline_thickness, 0.2 * values_rgb, values_rgb)
    else:
        values_rgb = 1.0 - dr.sign(values) * mi.Vector3f(0.3, 0.5, 0.65) * 1.3
        values_rgb = dr.clamp(values_rgb * 1.2, 0, 1)
        values_rgb *= 1.0 - 0.5 * dr.exp(-3.0 * dr.abs(values))
        values_rgb *= 0.8 + 0.2 * dr.cos(200.0 * values)
        values_rgb = values_rgb ** 1.1

    values_rgb = np.array(values_rgb).reshape(res, res, 3)
    border_size = 5
    edge_color = np.array([[[0.2, 0.5, 0.8]]]) * 0.25
    if use_bunny_scene:
        values_rgb[:border_size, :, :] = edge_color
        values_rgb[-border_size:, :, :] = edge_color
    else:
        values_rgb[:border_size * 2, :, :] = edge_color
        values_rgb[-border_size * 2:, :, :] = edge_color
    values_rgb[:, :border_size, :] = edge_color
    values_rgb[:, -border_size:, :] = edge_color

    if use_bunny_scene:
        values_rgb *= 0.4

    values_rgb = mi.TensorXf(mi.Float(values_rgb.ravel()), (res, res, 3))

    params[tex_key] = values_rgb
    params.update()

    print("[+] Rendering...")
    img = mi.render(sdf_scene, sensor=sensor, spp=spp)
    mi.Bitmap(img).write(join(output_dir, 'render.exr'))
    print("[+] Done rendering")


if __name__ == '__main__':
    render_sdf_isolines_figure()
