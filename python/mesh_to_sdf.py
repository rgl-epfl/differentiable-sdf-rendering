
import mitsuba as mi
import drjit as dr

import redistancing
import numpy as np


def create_sdf(mesh_fn, resolution, refine_surface=True):
    """Convert a watertight mesh to an SDF using ray tracing and fast sweeping"""

    mesh_plugin_type = "obj" if mesh_fn.endswith('.obj') else "ply"
    mesh_scene = mi.load_dict({"type": "scene", "integrator": {"type": "path"},
                            "sensor": {"type": "perspective"},
                            "myshape": {"type": mesh_plugin_type, "filename": mesh_fn}})
    res = resolution

    # Part I: Compute SDF by first tracing rays to determine a binary occupancy
    # mask Then use fast sweeping to extend that to an SDF. This assumes the shape is watertight.
    z, y, x = dr.meshgrid(*[dr.linspace(mi.Float, -0.5 + 0.5 / res, 0.5 - 0.5 / res, res)
                          for i in range(3)], indexing='ij')
    ray = mi.Ray3f(mi.Point3f(x, y, z), dr.normalize(mi.Vector3f(0, 1, 0)))
    si = mesh_scene.ray_intersect(ray)
    values = dr.select(si.is_valid() & (dr.dot(si.n, ray.d) > 0), 1.0, 0.0)
    values = 0.5 - values
    grid = mi.TensorXf(redistancing.redistance(mi.TensorXf(values, (res, res, res))))

    # Part II:  Given grid: for each voxel center next to the surface, localize
    # surface more precisely by tracing a number of rays to find the true
    # surface distance
    if refine_surface:
        # Gather voxels close to surface
        indices = dr.arange(mi.UInt32, res ** 3)
        near_surface_indices = mi.UInt32(np.array(indices)[dr.abs(grid.array) < 1.0 / res])

        # For each index, get the world space ray origin
        ray_o = dr.gather(mi.Point3f, ray.o, near_surface_indices)
        angular_res = 16
        n_angle_samples = angular_res ** 2
        r = dr.arange(mi.Float, angular_res)
        u, v = dr.meshgrid((r + 0.5) / angular_res, (r + 0.5) / angular_res)
        uv = dr.tile(mi.Vector2f(u, v), dr.width(ray_o))
        ray = mi.Ray3f(dr.repeat(ray_o, n_angle_samples), mi.warp.square_to_uniform_sphere(uv))

        # Trace these rays and find the minimum distance and modulate by sign
        si = mesh_scene.ray_intersect(ray)
        min_dist = dr.full(mi.Float, 100.0, dr.width(near_surface_indices))
        j = dr.arange(mi.UInt32, dr.width(near_surface_indices))
        i = mi.UInt32(0)
        loop = mi.Loop("BlockMin", lambda: (i, min_dist))
        while loop(i < n_angle_samples):
            min_dist = dr.minimum(min_dist, dr.gather(mi.Float, si.t, j * n_angle_samples + i))
            i += 1
        min_dist = min_dist * dr.sign(dr.gather(mi.Float, grid.array, near_surface_indices))
        dr.scatter(grid.array, min_dist, near_surface_indices)
        grid = redistancing.redistance(grid)
    return grid
