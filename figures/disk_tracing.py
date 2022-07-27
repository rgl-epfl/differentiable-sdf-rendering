"""This file implements a 2D version of our sphere tracing routine for
   visualization in paper figures"""

import drjit as dr
import numpy as np

import mitsuba as mi
from math_util import bbox_distance_inside


SDF_TRACE_EPSILON = 1e-6
SIL_WEIGHT_OFFSET = 0.05

def sphere_tracing_step_weight(ray, sdf_value, sdf_grad, bbox=None, p=None,
                               i=None, sil_weight_offset=SIL_WEIGHT_OFFSET):
    n = sdf_grad / dr.norm(sdf_grad)
    n_dot_d = dr.dot(ray.d, n)
    w = 1 / (1e-7 + dr.abs(sdf_value) + sil_weight_offset * n_dot_d ** 2) ** 3

    if bbox is not None:
        # Downweight as bbox is approached
        bbox_dist = bbox_distance_inside(p, bbox)
        bbox_eps = 0.02
        bbox_weight = dr.select(i > 0, dr.minimum(bbox_dist, bbox_eps) / bbox_eps, 1.0)
        w = w * bbox_weight
    return w

def intersect_sdf_simple(sdf, ray, use_approach_weighting=True, symbolic=False,
                         sil_weight_offset=SIL_WEIGHT_OFFSET):
    t = dr.zeros(mi.Float, dr.width(ray.d))
    points = []
    dists = []
    bbox = mi.BoundingBox2f(mi.Point2f(0, 0), mi.Point2f(1, 1))
    active = mi.Mask(True)
    weight_integral = mi.Float(0.0)
    warp_t_integral = mi.Float(0.0)
    prev_sdf_value = mi.Float(0.0)
    prev_sil_w = mi.Float(0.0)
    i = mi.UInt32(0)
    extra_weight_sum = mi.Float(0.0)

    if symbolic:
        loop = mi.Loop("Disk tracing", state=lambda: (t, i, active, weight_integral, warp_t_integral,
                                                       prev_sdf_value, prev_sil_w, extra_weight_sum))
    else:
        loop = lambda active: dr.any(active)

    while loop(active):
        current_p = ray(t)
        sdf_value = sdf.eval(current_p)
        surf_dist = dr.abs(sdf_value)
        intersected = active & (surf_dist < SDF_TRACE_EPSILON)
        sdf_grad = dr.detach(sdf.eval_grad(current_p))

        sil_w = sphere_tracing_step_weight(ray, sdf_value, sdf_grad, bbox, current_p, i, sil_weight_offset)
        sil_w = dr.maximum(sil_w, 0.0)
        segment_length = dr.maximum(prev_sdf_value, 0.0)
        prev_t = t - prev_sdf_value

        extra_weight_sum += dr.maximum(0.0, prev_sdf_value - surf_dist) / dr.minimum(0.05, sdf_value)
        extra_weight_sum = dr.clamp(extra_weight_sum, 0.0, 1.0)
        if use_approach_weighting:
            sil_w *= extra_weight_sum

        # Trapozoid rule
        weight_integral[active] = weight_integral + 0.5 * segment_length * (prev_sil_w + sil_w)
        warp_t_integral[active] = warp_t_integral + 0.5 * segment_length * (prev_t * prev_sil_w + t * sil_w)

        # Collect all points that were encountered
        if not symbolic:
            points.append(np.array((current_p.x, current_p.y)))
            dists.append(np.array(sdf_value))
        active &= ~intersected
        active &= bbox.contains(current_p)
        prev_sil_w[active] = sil_w
        prev_sdf_value[active] = sdf_value
        t[~intersected] += sdf_value
        i[active] += 1

    warp_t_integral = warp_t_integral / weight_integral
    return t, warp_t_integral, points, dists, weight_integral
