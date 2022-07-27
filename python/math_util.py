import mitsuba as mi
import drjit as dr


def normalize(x):
    """Normalizes a vector and additionally returns the Jacobian of the normalization"""
    x2 = dr.squared_norm(x)
    inv_v = dr.rsqrt(x2)
    jac = inv_v * mi.Matrix3f(1.0) - (inv_v / x2) * outer_product(x, x)
    return x * inv_v, jac


def normalize_sqr(x):
    """Normalizes a vector by its squared norm and returns the Jacobian of the normalization"""
    x2 = dr.squared_norm(x)
    jac = mi.Matrix3f(1.0) / x2 - (2 / dr.sqr(x2)) * outer_product(x, x)
    return x / x2, jac


def outer_product(v, w):
    """Computes the outer product between two 3D vectors"""
    return mi.Matrix3f(v.x * w.x, v.x * w.y, v.x * w.z,
                       v.y * w.x, v.y * w.y, v.y * w.z,
                       v.z * w.x, v.z * w.y, v.z * w.z)


def bbox_distance_inside(x, bbox):
    return dr.maximum(0.0, dr.minimum(dr.min(x - bbox.min), dr.min(bbox.max - x)))


def bbox_distance_inside_d(x, bbox):
    bbox_dist = dr.maximum(0.0, dr.minimum(dr.min(x - bbox.min), dr.min(bbox.max - x)))
    bbox_max_dist = dr.abs(bbox.max - x)
    bbox_min_dist = dr.abs(bbox.min - x)
    min_dist = dr.minimum(bbox_min_dist, bbox_max_dist)
    n = mi.Vector3f(0.0)
    n[(min_dist.x < min_dist.y) & (min_dist.x < min_dist.z)] = mi.Vector3f(1, 0, 0)
    n[(min_dist.y < min_dist.z) & (min_dist.y < min_dist.x)] = mi.Vector3f(0, 1, 0)
    n[(min_dist.z < min_dist.x) & (min_dist.z < min_dist.y)] = mi.Vector3f(0, 0, 1)
    bbox_dist_d = dr.select(bbox_dist > 0.0, n * dr.sign(bbox_max_dist - bbox_min_dist), 0.0)
    return bbox_dist, bbox_dist_d
