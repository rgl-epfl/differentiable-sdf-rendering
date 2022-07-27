import drjit as dr
import mitsuba as mi
import numpy as np

import math_util as util
import redistancing
from util import atleast_4d

XX = 0
YY = 1
ZZ = 2
XY = 3
XZ = 4
YZ = 5


def detach(x, detach_variable=True):
    return dr.detach(x) if detach_variable else x


def sigmoid(x):
    return dr.exp(x) / (1 + dr.exp(x))


class SDFBase:
    """Base class for 3D SDFs that defines our custom sphere tracing routine."""

    def __init__(self):
        # Slightly reduce SDF step scale by default to counteract grid interpolataion inaccuracies
        self.step_scale = 1.0
        self.trace_eps = 1e-6
        self.p = mi.Point3f(0.0)
        self.refine_intersection = True
        self.use_extra_weight = True
        self.extra_thresh = 0.05
        self.sil_weight_offset = 0.05
        self.sil_weight_epsilon = 1e-6
        self.weight_power = 3
        self.use_weight_sum_weight = True

        self.use_weight_ad = False

    def class_(self):
        class SDFClass:
            def name(self):
                return 'SDF'
        return SDFClass()

    def eval(self, p, detached=False):
        raise NotImplementedError

    def eval_and_grad(self, p, detached=False):
        raise NotImplementedError

    def eval_grad(self, x, detached=False):
        raise NotImplementedError

    def eval_all(self, x, detach_w=False):
        raise NotImplementedError

    def update(self):
        pass

    def bbox(self, expand=True):
        delta = 0.05 if expand else 0.0
        return mi.BoundingBox3f(dr.detach(self.p) - delta, dr.detach(self.p) + 1 + delta)

    def eval_trace_weight(self, ray, i, bbox, x, sdf_value, sdf_grad, hessian):

        # Distance + normalized dot product
        n_dot_d = dr.dot(sdf_grad, ray.d)
        n_dot_n = dr.dot(sdf_grad, sdf_grad)
        dot_ratio = n_dot_d / n_dot_n
        denom = self.sil_weight_epsilon + dr.abs(sdf_value) + self.sil_weight_offset * n_dot_d * dot_ratio
        dist_weight = 1 / denom ** self.weight_power

        # TODO: Does this weight help in some cases? Seems problematic in some cases as it is too aggressive
        # weight = weight * dr.exp(-sdf_value ** 2 / 0.01)

        # Downweight as bounding box is approached (dont do this for first iteration)
        if self.use_weight_ad:
            bbox_dist = util.bbox_distance_inside(x, bbox)
        else:
            bbox_dist, bbox_dist_d = util.bbox_distance_inside_d(x, bbox)

        bbox_eps = 0.01
        bbox_weight = dr.select(i > 0, dr.minimum(bbox_dist, bbox_eps) / bbox_eps, 1.0)
        weight = dist_weight * bbox_weight

        # Downweight influence of starting location
        # dist_from_last_interaction = dr.select((i > 0) | (mint > 0), dr.abs(t), 0.0)
        # weight = weight * dr.minimum(dist_from_last_interaction / 0.05, 1)

        # Downweight same orientation
        # front_facing_factor = -dr.dot(ray.d, sdf_grad)
        # weight *= sigmoid((front_facing_factor + 0.25) * 1)

        if not self.use_weight_ad:
            bbox_weight_d = dr.select((i > 0) & (bbox_dist < bbox_eps), bbox_dist_d / bbox_eps, 0.0)
            # gradient = 2 * ray.d * n_dot_d / n_dot_n - 2 * n_dot_d2 / dr.sqr(n_dot_n) * sdf_grad
            gradient = 2 * dot_ratio * (ray.d - dot_ratio * sdf_grad)
            denom_d = dr.sign(sdf_value) * sdf_grad + self.sil_weight_offset * gradient @ hessian
            dist_weight_d = -self.weight_power * dist_weight / denom * denom_d
            weight_d = dist_weight * bbox_weight_d + bbox_weight * dist_weight_d
        else:
            # Compute gradients of weight term
            # dr.set_grad(weight, 0.0)
            assert dr.grad_enabled(weight)
            dr.backward(weight)
            weight_d = dr.grad(x)
            weight = dr.detach(weight)

        return weight, weight_d

    def ray_intersect(self, ray, warp=None, active=True, extra_outputs=None):
        request_gradient = warp is not None
        if not request_gradient:  # faster, simpler code path
            return self.ray_intersect_non_diff(ray, active)

        loop_record_state = dr.flag(dr.JitFlag.LoopRecord)
        dr.set_flag(dr.JitFlag.LoopRecord, True)

        ray = mi.Ray3f(ray)
        ray.d = dr.normalize(ray.d)

        def convert_deriv(in_d, dist, dist_d):
            return dr.fma(dist, in_d, dr.dot(ray.d, in_d) * dist_d)

        bbox = self.bbox()
        intersects_bbox, mint, maxt = bbox.ray_intersect(ray)
        inside_bbox = bbox.contains(ray.o)
        intersects_bbox &= (mint > 0) | inside_bbox
        active = mi.Mask(active)
        active &= intersects_bbox

        ray.maxt = dr.minimum(maxt, ray.maxt)
        trace_eps = self.trace_eps * dr.maximum(ray.maxt, 1)

        scale = dr.opaque(mi.Float, self.step_scale) if self.step_scale != 1.0 else None
        its_t = mi.Float(dr.inf)
        t = dr.select(inside_bbox, 0.0, mint + 1e-5)
        warp_t = mi.Float(0.0)
        prev_surf_dist = mi.Float(0.0)
        prev_sdf_grad_c = mi.Vector3f(0.0)
        weight_sum = mi.Float(0.0)

        mixed_sum_d = mi.Vector3f(0.0)
        weight_d_sum = mi.Vector3f(0.0)
        i = mi.Int32(0)

        extra_weight_sum = mi.Float(0.0)
        extra_weight_sum_d = mi.Vector3f(0.0)
        use_extra_weight = self.use_extra_weight and request_gradient

        # Check which side of the bounding box is the closest
        bbox_its_p = ray(t)
        min_dist = dr.minimum(dr.abs(bbox.min - bbox_its_p), dr.abs(bbox.max - bbox_its_p))
        n = mi.Vector3f(0.0)
        n[(min_dist.x < min_dist.y) & (min_dist.x < min_dist.z)] = mi.Vector3f(1, 0, 0)
        n[(min_dist.y < min_dist.z) & (min_dist.y < min_dist.x)] = mi.Vector3f(0, 1, 0)
        n[(min_dist.z < min_dist.x) & (min_dist.z < min_dist.y)] = mi.Vector3f(0, 0, 1)
        d_dot_n = dr.dot(ray.d, n)
        t_d = mi.Vector3f(0.0)
        t_d[~inside_bbox & (dr.abs(d_dot_n) > 0)] = -n / d_dot_n * t

        loop = mi.Loop(name='SphereTracing',
                       state=lambda: (active, t, its_t, prev_surf_dist, prev_sdf_grad_c, t_d,
                                      weight_sum, weight_d_sum, mixed_sum_d, warp_t, i,
                                      extra_weight_sum, extra_weight_sum_d))
        while loop(active):
            x = ray(t)
            if self.use_weight_ad:
                dr.enable_grad(x)
                sdf_value, sdf_grad = self.eval_and_grad(x, True)
                hessian = None
            else:
                with dr.suspend_grad():
                    sdf_value, _, sdf_grad, _, hessian = self.eval_all(x)

            if scale is not None:
                sdf_value = scale * sdf_value
                sdf_grad = scale * sdf_grad
                hessian = mi.Float(scale) * hessian

            intersected = sdf_value < trace_eps
            its_t[intersected] = t
            surf_dist = dr.abs(sdf_value)
            weight, weight_d = self.eval_trace_weight(ray, i, bbox, x, sdf_value, sdf_grad, hessian)

            if self.use_weight_ad:
                sdf_value = dr.detach(sdf_value, True)
                sdf_grad = dr.detach(sdf_grad, True)
                surf_dist = dr.detach(surf_dist, True)

            if use_extra_weight:
                # Start increasing weight only as a new surface is approached
                # Increase quickly enough such that the weight -> 1 if a surface is near
                inv_extra_w_den = 1 / dr.minimum(self.extra_thresh, surf_dist)
                dist_difference = prev_surf_dist - surf_dist
                extra_weight_sum += dr.select(dist_difference >= 0, dist_difference * inv_extra_w_den, 0.0)
                extra_weight_sum = dr.minimum(extra_weight_sum, 1.0)

            curr_segment_value = dr.select(intersected, 0.0, surf_dist)
            segment_length = 0.5 * (curr_segment_value + prev_surf_dist)
            weight_increment = segment_length * weight
            if use_extra_weight:
                weight_increment *= extra_weight_sum
            weight_sum = weight_sum + weight_increment
            warp_t = warp_t + weight_increment * t

            # If we actually intersect, we need to update the t_d derivatives
            # t_d = dr.select(intersected, sdf_grad * t / dr.dot(sdf_grad, -ray.d), t_d)
            # Accumulate gradient terms

            weight_d = convert_deriv(weight_d, t, t_d)
            sdf_grad_c = convert_deriv(sdf_grad, t, t_d)
            segment_d = 0.5 * (sdf_grad_c + prev_sdf_grad_c)

            if use_extra_weight:
                sdf_sign = dr.sign(sdf_value)
                surf_dist_d = sdf_sign * sdf_grad_c
                extra_w_d = (prev_sdf_grad_c - surf_dist_d) * inv_extra_w_den
                extra_w_d = extra_w_d - dist_difference * \
                    dr.sqr(inv_extra_w_den) * dr.select(sdf_value < self.extra_thresh, surf_dist_d, 0.0)
                extra_weight_sum_d += dr.select(dist_difference > 0.0, extra_w_d, 0.0)
                extra_weight_sum_d[(extra_weight_sum >= 1.0) | (extra_weight_sum <= 0.0)] = 0.0
                weight_d = weight * extra_weight_sum_d + weight_d * extra_weight_sum
                weight *= extra_weight_sum

            weight_increment_d = dr.fma(weight, segment_d, weight_d * segment_length)
            mixed_sum_d += dr.fma(weight_increment_d, t, weight * segment_length * t_d)
            t_d = t_d + sdf_grad_c
            weight_d_sum += weight_increment_d
            i += 1
            t += curr_segment_value
            prev_surf_dist = surf_dist
            prev_sdf_grad_c = sdf_grad_c
            active &= (t <= ray.maxt) & (~intersected)

        if extra_outputs is not None:
            extra_outputs['i'] = i
            extra_outputs['weight_sum'] = weight_sum

        # Make intersection more precise by taking a few extra steps at a decreasing rate
        if self.refine_intersection:
            refining = mi.Mask(dr.isfinite(its_t))
            i = mi.Int32(0)
            loop = mi.Loop('SphereTracingRefine')
            loop.put(lambda: (refining, its_t, i))
            loop.init()
            while(loop(refining)):
                min_dist = dr.detach(self.eval(ray(its_t)))
                its_t[refining] += min_dist * (mi.Float(10) / mi.Float(10 + i))
                # TODO: make this agnostic to inside vs outside ray origins
                refining &= (min_dist <= 0) | (min_dist > trace_eps)
                i += 1
                refining &= i < 10

        inv_weight_sum = 1 / weight_sum
        warp_t = warp_t * inv_weight_sum
        warp_t_d = (-warp_t * weight_d_sum + mixed_sum_d) * inv_weight_sum
        # TODO Debug
        # warp_t = weight_sum
        # warp_t_d = weight_d_sum
        # DEBUG: Differentiate ray intersection distance wrt. ray direction
        # sdf_value, sdf_grad = self.eval_and_grad(ray(warp_t))
        # warp_t_d = sdf_grad * warp_t / dr.dot(sdf_grad, -ray.d)

        # Return weight sum as additional warp field weight multiplier
        if self.use_weight_sum_weight:
            warp_weight = dr.clamp(weight_sum, 0.0, 1.0)
            warp_weight_d = dr.select((weight_sum > 0.0) & (weight_sum < 1.0), weight_d_sum, 0.0)
        else:
            warp_weight = None
            warp_weight_d = None

        # Disable warp field for weight below some threshold
        invalid = (weight_sum < 1e-7) | ~intersects_bbox
        warp_t[invalid] = dr.inf
        warp_t_d[invalid] = 0.0
        if self.use_weight_sum_weight:
            warp_weight[invalid] = 0.0
            warp_weight_d[invalid] = 0.0

        # Potentially reset loop record flag
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record_state)

        return its_t, warp_t, warp_t_d, warp_weight, warp_weight_d

    def ray_intersect_non_diff(self, ray, active=True):
        loop_record_state = dr.flag(dr.JitFlag.LoopRecord)
        dr.set_flag(dr.JitFlag.LoopRecord, True)

        ray = mi.Ray3f(ray)
        ray.d = dr.normalize(ray.d)
        bbox = self.bbox()
        intersects_bbox, mint, maxt = bbox.ray_intersect(ray)
        inside_bbox = bbox.contains(ray.o)
        intersects_bbox &= (mint > 0) | inside_bbox
        active = mi.Mask(active)
        active &= intersects_bbox
        ray.maxt = dr.minimum(maxt, ray.maxt)
        scale = dr.opaque(mi.Float, self.step_scale) if self.step_scale != 1.0 else None
        its_t = mi.Float(dr.inf)
        t = dr.select(inside_bbox, 0.0, mint + 1e-5)

        trace_eps = self.trace_eps * dr.maximum(ray.maxt, 1)
        loop = mi.Loop(name='SphereTracingNonDiff',
                       state=lambda: (t, its_t, active))
        while loop(active):
            current_p = ray(t)
            sdf_value = self.eval(current_p)
            if scale is not None:
                sdf_value = scale * sdf_value
            intersected = sdf_value < trace_eps
            its_t = dr.select(intersected, t, its_t)
            curr_segment_value = dr.select(intersected, 0.0, dr.abs(sdf_value))
            active &= (t <= ray.maxt) & (~intersected)
            t += dr.detach(curr_segment_value)
            active &= t <= ray.maxt

        # Make intersection more precise by taking a few extra steps at a decreasing rate
        if self.refine_intersection:
            refining = mi.Mask(dr.isfinite(its_t))
            i = mi.Int32(0)
            loop = mi.Loop(name='SphereTracingRefineNonDiff',
                           state=lambda: (refining, its_t, i))
            while(loop(refining)):
                min_dist = dr.detach(self.eval(ray(its_t)))
                its_t[refining] = its_t + min_dist * (mi.Float(10) / mi.Float(10 + i))
                # TODO: make this agnostic to inside vs outside ray origins
                refining &= (min_dist <= 0) | (min_dist > trace_eps)
                i = i + 1
                refining &= i < 10

        # Potentially reset loop record flag
        dr.set_flag(dr.JitFlag.LoopRecord, loop_record_state)

        return its_t, mi.Float(0.0), mi.Vector3f(0.0), None, None

    def ray_intersect_preliminary(self, ray, active=True):
        its_t = self.ray_intersect_non_diff(ray, active)[0]
        pi = dr.zeros(mi.PreliminaryIntersection3f)
        pi.t = its_t
        return pi

    def compute_surface_interaction(self, ray, t):
        si = dr.zeros(mi.SurfaceInteraction3f)
        p = ray(t)

        # TODO: Detect if we need gradients here or not
        differentiable = True
        if differentiable:
            sdf_value, sdf_grad = self.eval_and_grad(p)
            t_diff = sdf_value / dr.detach(dr.dot(sdf_grad, -ray.d))
            t = dr.replace_grad(mi.Float(t), t_diff)
        si.t = t
        si.p = ray(t)
        si.sh_frame.n = dr.normalize(self.eval_grad(si.p))
        si.initialize_sh_frame()
        si.n = si.sh_frame.n
        si.wi = dr.select(si.is_valid(), si.to_local(-ray.d), -ray.d)
        si.wavelengths = ray.wavelengths
        si.dp_du = si.sh_frame.s
        si.dp_dv = si.sh_frame.t
        return si

    def parameters_changed(self, keys):
        return None

    def traverse(self):
        return


class Grid3d(SDFBase):
    """Grid-based SDF class. This is the class that is used for all our optimizations."""

    def __init__(self, data, transform=None):
        super().__init__()

        self.has_transform = transform is not None
        if transform is None:
            transform = mi.ScalarTransform4f(1.0)
        if type(data) is str:
            data = mi.Thread.thread().file_resolver().resolve(data)
            data = redistancing.redistance(mi.TensorXf(mi.VolumeGrid(data)))
        self.texture = mi.Texture3f(data.shape[:3], 1, use_accel=False)
        self.texture.set_tensor(atleast_4d(data), migrate=False)
        self.p = mi.Vector3f(0, 0, 0)
        self.to_world = transform
        self.to_local = self.to_world.inverse()
        self.update_bbox()

    def update_bbox(self):
        self.aabb = mi.ScalarBoundingBox3f()
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(0.0, 0.0, 0.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(0.0, 0.0, 1.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(0.0, 1.0, 0.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(0.0, 1.0, 1.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(1.0, 0.0, 0.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(1.0, 0.0, 1.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(1.0, 1.0, 0.0))
        self.aabb.expand(self.to_world @ mi.ScalarPoint3f(1.0, 1.0, 1.0))

    def resolution(self):
        return self.grid.resolution()

    def call_wrap(self, x, fn, detached=False):
        if detached:
            with dr.suspend_grad():
                with dr.resume_grad(x):
                    return fn(mi.Transform4f(self.to_local) @ (x - self.p))
        else:
            return fn(mi.Transform4f(self.to_local) @ (x - self.p))

    def bbox(self, expand=True):
        delta = 0.05 if expand else 0.0
        return mi.BoundingBox3f(self.aabb.min - delta, self.aabb.max + delta)

    def eval(self, x, detached=False):
        return self.call_wrap(x, self.texture.eval_cubic, detached)[0]

    def eval_grad(self, x, detached=False):
        """Evaluates the image gradient using bspline interpolation"""
        g = mi.Vector3f(self.call_wrap(x, self.texture.eval_cubic_grad, detached)[1][0])
        if self.has_transform:
            return mi.Transform4f(self.to_world) @ mi.Normal3f(g.x, g.y, g.z)
        return g

    def eval_and_grad(self, x, detached=False):
        """Evaluates the image gradient using bspline interpolation"""
        v, g = self.call_wrap(x, self.texture.eval_cubic_grad, detached)
        g = mi.Vector3f(g[0])
        if self.has_transform:
            g = mi.Transform4f(self.to_world) @ mi.Normal3f(g.x, g.y, g.z)
        return mi.Float(v[0]), g

    def eval_all(self, x, detach_w=False):
        """Evaluates the image hessian using bspline interpolation"""
        v, g, h = self.call_wrap(x, self.texture.eval_cubic_hessian, False)
        v = mi.Float(v[0])
        g = mi.Vector3f(g[0])
        h = mi.Matrix3f(h[0])

        if self.has_transform:
            to_local3 = mi.Matrix3f(self.to_local.matrix)
            g = mi.Vector3f(dr.transpose(to_local3) @ g)
            h = dr.transpose(to_local3) @ h @ to_local3

        return v, dr.detach(v, True), g, dr.detach(g, True), h

    def eval_all_detached(self, x, detach_w=False):
        """Evaluates the image hessian using bspline interpolation"""
        v, g, h = self.texture.eval_cubic_hessian(x - dr.detach(self.p))
        v = mi.Float(v[0])
        g = mi.Vector3f(g[0])
        h = mi.Matrix3f(h[0])

        if self.has_transform:
            to_local3 = mi.Matrix3f(self.to_local.matrix)
            g = mi.Vector3f(dr.transpose(to_local3) @ g)
            h = dr.transpose(to_local3) @ h @ to_local3

        return v, dr.detach(v, True), g, dr.detach(g, True), h

    def eval_1_hessian_d(self, x, detach_w=False):
        return mi.Matrix3f(self.call_wrap(x, self.texture.eval_cubic_hessian, False)[2][0])

    def traverse(self, callback):
        callback.put_parameter("sdf.data", self.texture.tensor(), mi.ParamFlags.Differentiable)
        callback.put_parameter("sdf.p", self.p, mi.ParamFlags.Differentiable)

    def update(self):
        self.update_bbox()
        self.texture.set_tensor(self.texture.tensor())

    def parameters_changed(self, keys):
        self.update_bbox()
        self.texture.set_tensor(self.texture.tensor())

    @property
    def shape(self):
        return self.texture.tensor().shape


class SphereSDF(SDFBase):
    """An SDF of a sphere, only used for testing."""

    def __init__(self, p, r):
        super().__init__()
        self.p = p
        self.r = r

    def eval(self, x, detached=False):
        return dr.norm(x - detach(self.p, detached)) - detach(self.r, detached)

    def eval_grad(self, x, detached=False):
        return dr.normalize(x - detach(self.p, detached))

    def eval_and_grad(self, x, detached=False):
        n = x - detach(self.p, detached)
        norm = dr.norm(n)
        return norm - detach(self.r, detached), n / norm

    def eval_hessian(self, x, detached=False):
        n = detach(self.p, detached) - x
        n2 = n * n
        tmp = dr.squared_norm(n)
        f = 1 / (tmp * dr.safe_sqrt(tmp))
        h = [f * (n2.y + n2.z), f * (n2.x + n2.z), f * (n2.x + n2.y), -n.x * n.y * f, -n.x * n.z * f, -n.y * n.z * f]
        h = mi.Matrix3f([[h[XX], h[XY], h[XZ]],
                         [h[XY], h[YY], h[YZ]],
                         [h[XZ], h[YZ], h[ZZ]]])
        return h

    def eval_all(self, x):
        h = self.eval_hessian(x, True)

        n = x - self.p
        norm = dr.norm(n)
        v = norm - self.r
        g = n / norm

        n = x - dr.detach(self.p)
        norm = dr.norm(n)
        v_d = norm - dr.detach(self.r)
        g_d = n / norm
        return v, v_d, g, g_d, h

    def bbox(self):
        delta = 0.05
        return mi.BoundingBox3f(dr.detach(self.p) - 0.5 - delta, dr.detach(self.p) + 0.5 + delta)

    def traverse(self, callback):
        callback.put_parameter("sdf.p", self.p)
        callback.put_parameter("sdf.r", self.r)


class BoxSDF(SDFBase):
    """Smooth box SDF from https://iquilezles.org/articles/distfunctions/"""

    def __init__(self, p, extents, smoothing=0.01):
        super().__init__()
        self.p = p
        self.extents = extents
        self.smoothing = smoothing

    def eval(self, x, detached=False):
        q = dr.abs(x - detach(self.p, detached)) - detach(self.extents, detached)
        return dr.norm(dr.maximum(q, 0.0)) + dr.minimum(dr.maximum(q.x, dr.maximum(q.y, q.z)), 0.0) - detach(self.smoothing, detached)

    def bbox(self):
        delta = 0.05
        return mi.BoundingBox3f(dr.detach(self.p) - 0.5 - delta, dr.detach(self.p) + 0.5 + delta)


def create_sphere_sdf(res, center=[0.5, 0.5, 0.5], radius=0.3, noise_sigma=0.0):
    z, y, x = np.meshgrid(np.linspace(0, 1, res[0]), np.linspace(
        0, 1, res[1]), np.linspace(0, 1, res[2]), indexing='ij')
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    if noise_sigma > 0:
        import scipy.ndimage

        noise_freq = 30
        # signed_dist = np.linalg.norm(pts - center, axis=-1) + np.sin(y.ravel() * noise_freq) * noise_sigma - radius

        if True:
            noise = np.random.randn(pts.shape[0]) * noise_sigma / 4
        else:
            # Blur noise a bit
            noise_sigma = noise_sigma * 35
            noise = np.random.randn(res[0], res[1], res[2]) * noise_sigma
            noise = scipy.ndimage.gaussian_filter(noise, 3.0)
            noise = noise.ravel()
            noise = noise + np.sin(y.ravel() * noise_freq) * 0.02
        signed_dist = np.linalg.norm(pts - center, axis=-1) + noise - radius
    else:
        signed_dist = np.linalg.norm(pts - center, axis=-1) - radius
    sdf = np.reshape(signed_dist, res).astype(np.float32)
    return redistancing.redistance(mi.TensorXf(sdf))


def create_block_sdf(resolution, center=[0.5, 0.5, 0.5]):
    r2 = resolution // 2
    signed_dist = np.ones([resolution] * 3)
    signed_dist[r2 - r2 // 6:r2 + r2 // 6, r2 - r2 // 6:r2 + r2 // 6, r2 - r2 // 2:r2 + r2 // 2] = -1
    # Process to make sure this is a valid sdf
    sdf = redistancing.redistance(mi.TensorXf(signed_dist.astype(np.float32)))
    return sdf
