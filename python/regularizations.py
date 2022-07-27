import drjit as dr
import mitsuba as mi


def eval_discrete_laplacian_reg(data, _=None):
    """Simple discrete laplacian regularization to encourage smooth surfaces"""

    def linear_idx(p):
        p.x = dr.clamp(p.x, 0, data.shape[0] - 1)
        p.y = dr.clamp(p.y, 0, data.shape[1] - 1)
        p.z = dr.clamp(p.z, 0, data.shape[2] - 1)
        return p.z * data.shape[1] * data.shape[0] + p.y * data.shape[0] + p.x

    shape = data.shape
    z, y, x = dr.meshgrid(*[dr.arange(mi.Float, shape[i]) for i in range(3)], indexing='ij')
    p = mi.Point3i(x, y, z)
    c = dr.gather(mi.Float, data.array, linear_idx(p))
    vx0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(-1, 0, 0)))
    vx1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(1, 0, 0)))
    vy0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, -1, 0)))
    vy1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 1, 0)))
    vz0 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, -1)))
    vz1 = dr.gather(mi.Float, data.array, linear_idx(p + mi.Vector3i(0, 0, 1)))
    laplacian = dr.sqr(c - (vx0 + vx1 + vy0 + vy1 + vz0 + vz1) / 6)
    return dr.sum(laplacian)
