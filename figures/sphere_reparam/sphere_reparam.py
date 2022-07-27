"""Figure 2: Visualization of reparameterization on the sphere.

This script is quite complex, as it first creates a virtual scene
that is rendered from a spherical sensor. The output of
that first pass is then used a texture on a sphere for the final visualization.
"""
import sys
import os

sys.path.append(os.path.abspath('../'))
from common import *

import spherical_sensor # import to register the spherical sensor

import mesh_to_sdf
import shapes

from constants import SCENE_DIR, FIGURE_DIR, SDF_DEFAULT_KEY_P

import numpy as np
import mitsuba as mi
import configs # needed to activate integrators

from integrators.reparam import ReparamIntegrator

import tqdm

FIG_NAME = 'sphere_reparam'

def set_params(scene, setter_dict):
    params = mi.traverse(scene)
    for k, v in setter_dict.items():
        params[k] = v
    params.update()

def create_arrows(origins, directions, fast=False, cylinder_radius=0.01, cone_factor=0.35):
    """Create mesh arrows for a given vector field. Fast=true enables a vectorized, but less accurate version"""
    import open3d # pip install open3d
    cone_radius = 1.8 * cylinder_radius
    if fast:
        length = dr.norm(directions)
        mean_length = np.mean(length)
        cylinder_height = (1 - cone_factor) * mean_length
        cone_height = cone_factor * mean_length
        arrow = open3d.geometry.TriangleMesh.create_arrow(cylinder_radius, cone_radius, cylinder_height, cone_height)
        v = mi.Point3f(np.asarray(arrow.vertices))
        f = mi.Point3i(np.asarray(arrow.triangles))
        n_verts = int(dr.width(v))
        n_faces = int(dr.width(f))
        n_arrows = int(dr.width(directions))
        v = dr.tile(v, n_arrows)
        f = dr.tile(f, n_arrows)
        offsets = dr.repeat(dr.arange(mi.UInt32, n_arrows), n_faces) * n_verts
        f = f + offsets
        d = directions
        d = dr.repeat(d, n_verts)
        T = mi.Transform4f.look_at([0, 0, 0], d, mi.Frame3f(d).s)
        v = T @ v + dr.repeat(origins, n_verts)
        return v, f
    else:
        origins = np.array(origins)
        directions = np.array(directions)
        vertices = []
        faces = []
        tri_offset = 0
        for o, d in zip(origins, directions):
            length = np.linalg.norm(d)
            cylinder_height = (1 - cone_factor) * length
            cone_height = cone_factor * length
            arrow = open3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius, cone_radius, cylinder_height, cone_height)
            d = mi.ScalarVector3f(d[0], d[1], d[2])
            T = mi.Transform4f.look_at([0, 0, 0], d, np.array(mi.Frame3f(d).s)[0])
            v = mi.Point3f(np.asarray(arrow.vertices))
            v = T @ v + o
            v = np.array(v)
            vertices.append(v)
            faces.append(np.asarray(arrow.triangles) + tri_offset)
            tri_offset += v.shape[0]
        return mi.Point3f(np.concatenate(vertices, axis=0)), mi.Point3u(np.concatenate(faces, axis=0))


def create_vector_field_mesh(origins, directions, scale=0.2):
    """Creates and writes out a mesh for a given vector field"""
    from constants import OUTPUT_DIR

    origins = np.array(origins)
    directions = np.array(directions)
    # Subsample input if 2D
    if len(origins.shape) == 3:
        f = origins.shape[0] // 64
        origins = np.array(origins[::f, ::f, :])
        directions = np.array(directions[::f, ::f, :])
        origins = np.reshape(origins, (-1, 3))
        directions = np.reshape(directions, (-1, 3))

    # Select only sufficiently long vectors to draw
    valid = np.array(dr.norm(mi.Vector3f(directions)) > 0.1)
    directions = directions[valid]
    origins = origins[valid]
    # origins = dr.normalize(mi.Vector3f(origins)) # Make sure origins are on the sphere

    directions = mi.Vector3f(directions)
    directions = directions * scale
    try:
        import open3d
        arrows_v, arrows_f = create_arrows(origins, directions, fast=False, cylinder_radius=0.005)
    except ImportError:
        print("Could not import open3d, cannot plot arrows")
        arrows_v = mi.Point3f([100, 100, 100])
        arrows_f = mi.Point3u([0, 1, 2])

    mesh = mi.Mesh("MyMesh", int(dr.width(arrows_v)), int(dr.width(arrows_f)))
    params = mi.traverse(mesh)
    params['vertex_positions'] = dr.ravel(arrows_v)
    params['faces'] = dr.ravel(arrows_f)
    tmp_dir = os.path.join(OUTPUT_DIR, 'figures', 'tmp', 'sphere_reparam')
    os.makedirs(tmp_dir, exist_ok=True)
    mesh.write_ply(join(tmp_dir, 'arrows.ply'))


class Silhouette2Integrator(ReparamIntegrator):
    """Integrator used to render spherical silhouettes for this figure"""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.background_alpha = props.get('background_alpha', 0.2)
        self.reparam_splat = props.get('reparam_splat', True)
        self.reparametrize = props.get('reparamtrize', True)
        self.textured = props.get('textured', False)

    def sample(self, mode, scene, sampler, ray,
               δL,  state_in, reparam, active, **kwargs):
        si, _, det, extra_output = self.ray_intersect(scene, sampler, ray, reparam=self.reparametrize)
        result = dr.select(active, si.emitter(scene, active).eval(si, active), 0.0)

        if self.textured:
            result[si.is_valid()] = mi.Float(1.0) * dr.abs(dr.dot(si.sh_frame.n, -ray.d))
        else:
            result[si.is_valid()] = mi.Float(1.0) * dr.maximum(dr.dot(si.sh_frame.n, dr.normalize(mi.Vector3f(-1,1,-1))), 0.0) + 0.5
            # result[si.is_valid()] = dr.sqr(si.p.y) + 0.3

        # Evaluate a procedurale "texture" on the shape
        if self.textured:
            result[si.is_valid()] *= dr.clamp((dr.abs(dr.cos(ray.d.x * 50)) + 0.5) *
                                              (dr.abs(dr.sin((ray.d.y + ray.d.z) * 80)) + 0.5), 0.0, 0.95)
        result *= dr.replace_grad(mi.Float(1.0), det)
        aovs = [dr.select(si.is_valid(), 1.0, 0.0), mi.Float(ray.d.x),
                mi.Float(ray.d.y), mi.Float(ray.d.z), mi.Float(det)]

        if not self.reparam_splat:
            ray.assign(dr.detach(ray, True))
            det = mi.Float(1.0)
        return mi.Spectrum(result), mi.Mask(True), det, aovs

    def aov_names(self):
        return ['mask', 'reparam.d.x', 'reparam.d.y', 'reparam.d.z', 'div']

    def to_string(self):
        return 'Silhouette2Integrator'


mi.register_integrator("silhouette2", lambda props: Silhouette2Integrator(props))


def get_sphere_vertical_alpha_gradient(resx, resy, thresh=0.55):
    thresh = 0.55
    y_values = np.clip(1 - 4 * np.linspace(0, 1, resy - int(resy * thresh)), 0, 1)
    alpha = np.ones((resy, resx))
    alpha[int(resy * thresh):, :] *= (y_values[:, None]) ** 2
    return alpha


def get_outline(silhouette, width):
    import skfmm
    silhouette = np.array(silhouette)
    a = skfmm.distance(0.5 - silhouette, dx=1 / silhouette.shape[0])
    return np.abs(a) < width


def img_tensor(data):
    return mi.TensorXf(np.atleast_3d(data))


def render_sphere_reparam_figure():
    render_divergence_term = False
    pbar = tqdm.tqdm(range(5 if render_divergence_term else 4))

    output_dir = join(FIGURE_DIR, FIG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    mesh_fn = join(SCENE_DIR, 'meshes', 'blob.obj')

    sdf = mesh_to_sdf.create_sdf(mesh_fn, 256)
    sdf_shape = shapes.Grid3d(sdf)

    verbose = False
    res_scale = 2
    resx = int(512 * res_scale)
    resy = int(512 * res_scale)
    sphere_resx = int(1024 * res_scale)
    sphere_resy = int(512 * res_scale)
    spp = 512
    sphere_spp = 16

    offset = mi.ScalarVector3f(0.0, -0.12, 0.0)

    # Render original SDF
    ref_scene_name = join(SCENE_DIR, 'figures', 'sphere_reparam', 'sphere_reparam.xml')
    spherical_scene_name = join(SCENE_DIR, 'figures', 'sphere_reparam', 'sphere_reparam_scene.xml')
    arrow_scene_name = join(SCENE_DIR, 'figures', 'sphere_reparam', 'sphere_reparam_arrows.xml')

    sdf_scene = mi.load_file(spherical_scene_name, resx=sphere_resx, resy=sphere_resy)
    sdf_scene.integrator().sdf = sdf_shape
    img = mi.render(sdf_scene, spp=sphere_spp)
    img = np.array(img)

    # Render SDF with slightly offset position
    sdf_shape = shapes.Grid3d(sdf, mi.ScalarTransform4f.translate(offset))
    sdf_scene = mi.load_file(spherical_scene_name, resx=sphere_resx, resy=sphere_resy)
    sdf_scene.integrator().sdf = sdf_shape
    img2 = mi.render(sdf_scene, spp=sphere_spp)
    img2 = np.array(img2)

    # Form final texture as combination of these two images
    fg_alpha = img[..., 4][..., None]

    obj_mask = np.clip(img[..., 4] + img2[..., 4], 0, 1)
    img_blended = img[..., :3] + (1 - fg_alpha) * 0.5 * img2[..., :3] * img2[..., 4][..., None]

    draw_outline = False
    if draw_outline:
        outline = get_outline(img[..., 4], 0.001)
        obj_mask += outline
        img_blended = img_blended * (1 - outline)[..., None]

    img_blended = np.clip(img_blended, 0.0, 1.0)

    # Make background slightly transparent
    alpha = np.clip(obj_mask + 0.2, 0, 1)
    alpha *= get_sphere_vertical_alpha_gradient(sphere_resx, sphere_resy)
    alpha = np.stack([np.clip(alpha, 0.0, 1.0), ] * 3, axis=-1)

    # Reduce alpha outside the objects
    alpha = 0.75 * alpha * (1 - obj_mask[..., None]) + alpha * obj_mask[..., None]

    if verbose:
        plt.figure()
        plt.imshow(alpha)
        plt.figure()
        plt.imshow(img_blended)
        plt.figure()
        plt.imshow(obj_mask)


    sphere_scene = mi.load_file(ref_scene_name, resx=resx, resy=resy)
    ref_scene_tex_key = "Sphere.bsdf.nested_bsdf.brdf_0.diffuse_reflectance.data"
    ref_scene_opacity_key = "Sphere.bsdf.opacity.data"
    set_params(sphere_scene, {ref_scene_tex_key: img_tensor(img_blended),
                              ref_scene_opacity_key: img_tensor(alpha)})
    img_comp = mi.render(sphere_scene, spp=spp)
    mi.util.write_bitmap(join(output_dir, 'img_0.exr'), img_comp[..., :4])
    pbar.update()

    # Compute warp field: For each ray, we now just accumualate it's reparametrized direction
    # Then, forward gradient gives the desired motion vectors
    sdf_scene = mi.load_file(spherical_scene_name, resx=sphere_resx, resy=sphere_resy)
    sdf_scene.integrator().reparam_splat = False
    sdf_scene.integrator().sdf = sdf_shape
    config = configs.Warp()
    warp = config.get_warpfield(sdf_shape)
    warp.edge_eps = 0.2
    sdf_scene.integrator().warp_field = warp

    params = mi.traverse(sdf_scene)
    param = params[SDF_DEFAULT_KEY_P].y
    dr.enable_grad(param)
    dr.set_grad(param, 0.0)
    img_reparam = mi.render(sdf_scene, params=params, spp=sphere_spp)
    img_reparam = img_reparam[..., 5:5 + 3]

    dr.forward(param)
    input_ray_directions = np.array(img_reparam)
    warp_field = np.array(dr.grad(img_reparam))

    if verbose:
        plt.figure()
        plt.imshow(np.concatenate([input_ray_directions, warp_field], axis=1))

    # Create an actual mesh of warp field direction arrows
    create_vector_field_mesh(input_ray_directions, warp_field)

    # Render sphere with arrow mesh
    sphere_scene = mi.load_file(arrow_scene_name, resx=resx, resy=resy)

    # Recompute alpha
    alpha = np.clip(img[..., 4] + 0.2, 0, 1)
    alpha *= get_sphere_vertical_alpha_gradient(sphere_resx, sphere_resy)
    alpha = np.stack([np.clip(alpha, 0.0, 1.0), ] * 3, axis=-1)
    set_params(sphere_scene, {ref_scene_tex_key: img_tensor(np.clip(img[..., :3], 0, 1)), ref_scene_opacity_key: img_tensor(alpha * 0.75)})

    img_arrows = mi.render(sphere_scene, spp=spp)
    mi.util.write_bitmap(join(output_dir, 'img_1.exr'), img_arrows[..., :4])
    pbar.update()

    # Get reparam integrand: Now also plot the reparametrized integrand itself
    sdf_scene = mi.load_file(spherical_scene_name, resx=sphere_resx, resy=sphere_resy)
    sdf_scene.integrator().sdf = sdf_shape
    sdf_scene.integrator().reparam_splat = False
    config = configs.Warp()
    warp = config.get_warpfield(sdf_shape)
    warp.edge_eps = 0.1
    sdf_scene.integrator().warp_field = warp

    params = mi.traverse(sdf_scene)
    param = params[SDF_DEFAULT_KEY_P].y

    dr.enable_grad(param)
    dr.set_grad(param, 0.0)
    img_reparam = mi.render(sdf_scene, params=params, spp=sphere_spp)
    divergence = img_reparam[..., 8]
    img_reparam = img_reparam[..., :3]
    img_reparam = (img_reparam[..., 0] + img_reparam[..., 1] + img_reparam[..., 2]) / 3
    dr.forward(param)

    integrand_reparam = np.array(dr.grad(img_reparam))
    divergence_grad = np.array(dr.grad(divergence))

    # Gradient integrand
    sphere_scene = mi.load_file(ref_scene_name, resx=resx, resy=resy)
    r = 10
    tex = apply_color_map(integrand_reparam, vmin=-r, vmax=r) ** 2.2
    alpha = get_sphere_vertical_alpha_gradient(sphere_resx, sphere_resy)
    alpha = np.stack([alpha, ] * 3, axis=-1)
    set_params(sphere_scene, {ref_scene_tex_key: img_tensor(tex), ref_scene_opacity_key: img_tensor(alpha)})
    img = mi.render(sphere_scene, spp=spp)
    mi.util.write_bitmap(join(output_dir, 'img_3.exr'), img)
    pbar.update()

    # Divergence term
    if render_divergence_term:
        sphere_scene = mi.load_file(ref_scene_name, resx=resx, resy=resy)
        tex = apply_color_map(divergence_grad, vmin=-r, vmax=r) ** 2.2
        alpha = get_sphere_vertical_alpha_gradient(sphere_resx, sphere_resy)
        alpha = np.stack([alpha, ] * 3, axis=-1)
        set_params(sphere_scene, {ref_scene_tex_key: img_tensor(tex), ref_scene_opacity_key: img_tensor(alpha)})
        img = mi.render(sphere_scene, spp=spp)
        mi.util.write_bitmap(join(output_dir, 'img_2.exr'), img)
        pbar.update()

    # Render non reparametrized gradient
    r = 10
    sdf_scene = mi.load_file(spherical_scene_name, resx=sphere_resx, resy=sphere_resy)
    sdf_scene.integrator().sdf = sdf_shape
    sdf_scene.integrator().reparam_splat = False
    sdf_scene.integrator().reparametrize = False
    config = configs.Warp()
    warp = config.get_warpfield(sdf_shape)
    warp.edge_eps = 0.1
    sdf_scene.integrator().warp_field = warp

    params = mi.traverse(sdf_scene)
    param = params[SDF_DEFAULT_KEY_P].y
    dr.enable_grad(param)
    dr.set_grad(param, 0.0)
    img = mi.render(sdf_scene, params=params, spp=sphere_spp)
    img = (img[..., 0] + img[..., 1] + img[..., 2]) / 3
    dr.forward(param)
    integrand_non_reparam = np.array(dr.grad(img))

    # Gradient integrand
    sphere_scene = mi.load_file(ref_scene_name, resx=resx, resy=resy)
    tex = apply_color_map(integrand_non_reparam, vmin=-r, vmax=r) ** 2.2
    alpha = get_sphere_vertical_alpha_gradient(sphere_resx, sphere_resy)
    alpha = np.stack([alpha, ] * 3, axis=-1)
    set_params(sphere_scene, {ref_scene_tex_key: img_tensor(tex), ref_scene_opacity_key: img_tensor(alpha)})
    img = mi.render(sphere_scene, spp=spp)
    mi.util.write_bitmap(join(output_dir, 'img_4.exr'), img)
    pbar.update()


if __name__ == '__main__':
    render_sphere_reparam_figure()
