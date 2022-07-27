"""This file contains some 2D SDFs that are used for the 2D illustrations in the paper."""

import numpy as np

import drjit as dr
import mitsuba as mi


def detach(x, detach_variable=True):
    return dr.detach(x) if detach_variable else x


class SDFBase:

    def __init__(self):
        self.step_scale = 1.0
        self.trace_eps = 1e-4
        self.use_weight_ad = True
        self.warpt_normal_offset = dr.opaque(mi.Float, 0.1)


class Grid2d(SDFBase):

    def __init__(self, data):
        super().__init__()
        self.shape = mi.Vector2i(np.array(data.shape))
        self.data = mi.Float(data.ravel())
        self.p = mi.Vector2f(0, 0)
        self.texture = mi.Texture2f(data.shape, 1, use_accel=False)
        self.texture.set_tensor(data[..., None])

    def eval(self, x, detached=False):
        """Queries a 2D image using bspline interpolation"""
        x = x - detach(self.p, detached)
        if detached:
            with dr.suspend_grad():
                with dr.resume_grad(x):
                    return self.texture.eval_cubic(x, force_drjit=True)[0]
        return self.texture.eval_cubic(x, force_drjit=True)[0]

    def eval_grad(self, x, detached=False):
        """Evaluates the image gradient using bspline interpolation"""
        x = x - detach(self.p, detached)
        if detached:
            with dr.suspend_grad():
                with dr.resume_grad(x):
                    return self.texture.eval_cubic_grad(x)[1][0]

        return self.texture.eval_cubic_grad(x)[1][0]

    def eval_hessian(self, x, detached=False):
        """Evaluates the image hessian using bspline interpolation"""
        x = x - dr.detach(self.p)
        assert not detached
        return self.texture.eval_cubic_hessian(x)[2][0]

    def traverse(self, callback):
        if self.texture is not None:
            callback.put_parameter("sdf.data", self.texture.tensor(), mi.ParamFlags.Differentiable)
        else:
            callback.put_parameter("sdf.data", self.data, mi.ParamFlags.Differentiable)

        callback.put_parameter("sdf.p", self.p, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        if self.texture is not None:
            self.texture.set_tensor(self.texture.tensor())


class DiskSDF(SDFBase):

    def __init__(self, p, r):
        super().__init__()
        self.p = p
        self.r = r

    def eval(self, x, detached=False):
        if detached:
            return dr.norm(x - dr.detach(self.p)) - dr.detach(self.r)

        return dr.norm(x - self.p) - self.r

    def eval_grad(self, x, detached=False):
        if detached:
            return dr.normalize(x - dr.detach(self.p))

        return dr.normalize(x - self.p)

    def eval_hessian(self, x, detached=False):
        v = x - (self.p if not detached else dr.detach(self.p))
        n = dr.norm(v)
        g = mi.Vector3f(0.0, 0.0, 0.0)

        g.x = 1 / n - 1 / (n * n * n) * v.x * v.x
        g.y = 1 / n - 1 / (n * n * n) * v.y * v.y
        g.z = -1 / (n * n * n) * v.x * v.y
        return mi.Matrix2f([[g.x, g.z], [g.z, g.y]])


class RectangleSDF(SDFBase):

    def __init__(self, p, extents, rotation_angle=0, offset=0.015):
        super().__init__()
        self.p = p
        self.extents = extents

        # TODO: Support offset and rotation
        # Rotation: rotate points before eval, rotate gradient vector
        self.rotation_angle = rotation_angle
        self.offset = offset

    def eval(self, x, detached=False):
        x = x - detach(self.p, detached)
        d = dr.abs(x) - detach(self.extents, detached)
        return dr.norm(dr.maximum(d, 0.0)) + dr.minimum(dr.maximum(d.x, d.y), 0.0) - self.offset

    def eval_grad(self, x, detached=False):
        x = x - detach(self.p, detached)
        w = dr.abs(x) - detach(self.extents, detached)
        s = mi.Vector2f(dr.select(x.x < 0.0, -1, 1), dr.select(x.y < 0.0, -1, 1))
        g = dr.maximum(w.x, w.y)
        q = dr.maximum(w, 0.0)
        l = dr.norm(q)
        inner = dr.select(w.x > w.y, mi.Vector2f(1, 0), mi.Vector2f(0, 1))
        return s * dr.select(g > 0.0, q / l, inner)

    def eval_hessian(self, x, detached=False):
        return mi.Matrix2f(0.0)


class UnionSDF(SDFBase):
    """Allows to add to different SDFs to the same scene"""

    def __init__(self, sdf1, sdf2, smooth=True, k=32):
        super().__init__()
        self.sdf1 = sdf1
        self.sdf2 = sdf2
        self.smooth = smooth
        self.k = k

    def eval(self, x, detached=False):
        v1 = self.sdf1.eval(x, detached)
        v2 = self.sdf2.eval(x, detached)

        if self.smooth:
            return -dr.log(dr.exp(-self.k * v1) + dr.exp(-self.k * v2)) / self.k
        else:
            return dr.select(v1 < v2, v1, v2)

    def eval_grad(self, x, detached=False):
        v1 = self.sdf1.eval(x, detached)
        v2 = self.sdf2.eval(x, detached)
        g1 = self.sdf1.eval_grad(x, detached)
        g2 = self.sdf2.eval_grad(x, detached)

        if self.smooth:
            k = self.k
            x0 = dr.exp(-k * v1)
            x1 = k * x0
            x2 = dr.exp(-k * v2)
            x3 = k * x2
            x4 = 1 / (k * (x0 + x2) + 1e-7)  # TODO: Is this epsilon here at the right place?
            return mi.Vector2f(-x4 * (-g1.x * x1 - g2.x * x3), -x4 * (-g1.y * x1 - g2.y * x3))
        else:
            return dr.select(v1 < v2, g1, g2)

    def eval_hessian(self, x, detached=False):
        v1 = self.sdf1.eval(x, detached)
        v2 = self.sdf2.eval(x, detached)
        h1 = self.sdf1.eval_hessian(x, detached)
        h2 = self.sdf2.eval_hessian(x, detached)
        h1 = mi.Vector3f(h1[0, 0], h1[1, 1], h1[0, 1])
        h2 = mi.Vector3f(h2[0, 0], h2[1, 1], h2[0, 1])
        if self.smooth:
            k = self.k
            g1 = self.sdf1.eval_grad(x, detached)
            g2 = self.sdf2.eval_grad(x, detached)
            x0 = dr.exp(-k * v1)
            x1 = k * x0
            x2 = g1.x * x1
            x3 = dr.exp(-k * v2)
            x4 = k * x3
            x5 = g2.x * x4
            x6 = k ** (-1)
            x7 = x0 + x3 + 1e-7
            x8 = x6 / x7 ** 2
            x9 = x8 * (x2 + x5)
            x10 = k ** 2
            x11 = x0 * x10
            x12 = x10 * x3
            x13 = x6 / x7
            x14 = g1.y * x1
            x15 = g2.y * x4
            x16 = -x14 - x15
            h = mi.Vector3f(-x13 * (g1.x ** 2 * x11 - h1.x * x1 + g2.x ** 2 * x12 - h2.x * x4) - x9 * (-x2 - x5),
                            -x13 * (g1.y ** 2 * x11 - h1.y * x1 + g2.y ** 2 * x12 - h2.y * x4) - x16 * x8 * (x14 + x15),
                            -x13 * (g1.x * g1.y * x11 - h1.z * x1 + g2.x * g2.y * x12 - h2.z * x4) - x16 * x9)
        else:
            h = dr.select(v1 < v2, h1, h2)
        return mi.Matrix2f([[h.x, h.z], [h.z, h.y]])


class HalfSpaceSDF(SDFBase):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def eval(self, x, detached=False):
        if detached:
            return (x.x - dr.detach(self.p.x))
        return (x.x - self.p.x)

    def eval_grad(self, x, detached=False):
        return mi.Vector2f(1.0, 0.0)

    def eval_hessian(self, x, detached=False):
        return mi.Vector3f(0.0, 0.0, 0.0)


def arc_sdf(p, theta=np.pi / 4, phi=-0.4, ra=0.25, rb=0.1):
    sca = np.array([np.cos(theta), np.sin(theta)])
    scb = np.array([np.cos(phi), np.sin(phi)])
    R = np.array([[sca[0], sca[1]], [-sca[1], sca[0]]])
    p = p @ R
    p[:, 0] = np.abs(p[:, 0])
    length = np.sqrt(np.sum(p ** 2, axis=1))
    dotprod = p[:, 0] * scb[0] + p[:, 1] * scb[1]
    k = np.where(scb[1] * p[:, 0] > scb[0] * p[:, 1], dotprod, length)
    return np.sqrt(length ** 2 + ra ** 2 - 2.0 * ra * k) - rb


def disk_sdf(p, r=0.25):
    return np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - r
