import drjit as dr
import mitsuba as mi

from math_util import outer_product, normalize_sqr, bbox_distance_inside_d


class WarpField2D:
    """Normal aligned warp field with analytic divergence computation. Supports distance based downweighting of the warp magnitude."""

    def __init__(self, sdf, weight_strategy=4, edge_eps=0.05):
        self.max_reparam_depth = -1
        self.sdf = sdf
        self.edge_eps = dr.opaque(mi.Float, edge_eps)
        self.weight_strategy = weight_strategy
        self.directional = True
        self.clamping_thresh = 0.0
        self.return_aovs = False

        # If true, counter act SDF inaccuracies by explicit normalization of warp field
        self.normalize_warp_field = True

        if self.weight_strategy == 2:
            self.edge_eps /= 4

    def weight(self, x, d, sdf_value, sdf_grad, edge_eps):
        edge_eps_d = mi.Float(0.0)
        bbox = self.sdf.bbox()
        bbox_dist, bbox_dist_d = bbox_distance_inside_d(x, bbox)
        use_edge_eps = edge_eps <= bbox_dist
        edge_eps_d = dr.select(use_edge_eps, mi.Vector3f(0.0), bbox_dist_d)
        edge_eps = dr.minimum(edge_eps, bbox_dist)
        inv_edge_eps = 1 / edge_eps
        surf_dist = dr.abs(sdf_value)
        fac = 1 - surf_dist * inv_edge_eps
        w = dr.maximum(fac, 0.0)
        w_d = -dr.sign(sdf_value) * sdf_grad * inv_edge_eps + surf_dist * dr.sqr(inv_edge_eps) * edge_eps_d
        w_d = dr.select(fac >= 0.0, w_d, 0.0)
        edge_eps_d = dr.select(use_edge_eps & (fac >= 0), surf_dist * dr.sqr(inv_edge_eps), 0.0)
        return w, w_d, edge_eps_d

    def weight_epsilon(self, t):
        if self.weight_strategy == 6:
            return self.edge_eps * dr.detach(t)
        else:
            return self.edge_eps

    def eval(self, x, ray_d, t, dt_dx, active=True, rng=None,
             extra_output=None, warp_weight=None, warp_weight_d=None,
             detach_extra_outputs=True):

        active = mi.Mask(active)
        active &= dr.isfinite(t)
        sdf_value, _, sdf_normal, sdf_normal_d, h_mat = self.sdf.eval_all(x)
        h_mat = dr.detach(h_mat, True)

        if self.normalize_warp_field:
            sdf_normal_d_n, norm_jac = normalize_sqr(sdf_normal_d)
            warp = -sdf_normal_d_n * sdf_value
            jac = -norm_jac @ h_mat * sdf_value - outer_product(sdf_normal_d_n, sdf_normal)
        else:
            sdf_normal_d_n = mi.Vector3f(sdf_normal_d)
            warp = -sdf_normal_d_n * sdf_value
            jac = -h_mat * sdf_value - outer_product(sdf_normal_d_n, sdf_normal)

        # Apply weighting of the warp field itself
        x = dr.detach(x, True)
        d = dr.detach(ray_d, True)
        weight, weight_grad, edge_eps_grad = self.weight(x, d, dr.detach(sdf_value),
                                                         dr.detach(sdf_normal), self.weight_epsilon(t))
        weight_grad += edge_eps_grad * ray_d * self.edge_eps
        if warp_weight is not None:
            assert warp_weight_d is not None
            weight_grad = weight_grad * warp_weight + weight * warp_weight_d
            weight *= warp_weight

        weight = dr.detach(weight, True)
        jac = outer_product(warp, weight_grad) + weight * jac
        warp = warp * weight

        # We simplified: normalize(ray.o + ray.d*warp_t + warp - detach(warp) - ray.o) = normalize(ray_d * t + warp - detach(warp))
        warp = dr.replace_grad(mi.Vector3f(0.0), warp)
        warp = ray_d * dr.maximum(self.clamping_thresh, t) + warp
        warp = dr.normalize(warp)
        # proj_jac = (mi.Matrix3f(1.0) - outer_product(warp, warp)) @ jac
        # Here it's fine to not attach the projection Jacobian (i.e. we use ray_d instead of warp to compute it)
        proj_jac = (mi.Matrix3f(1.0) - outer_product(ray_d, ray_d)) @ jac
        jac = proj_jac + proj_jac @ outer_product(ray_d, dt_dx / t)
        div = jac[0, 0] + jac[1, 1] + jac[2, 2]


        active &= weight > 0
        div = dr.select(active, div, 0.0)
        warp = dr.select(active, warp, ray_d)
        ray_d = mi.Vector3f(ray_d)
        warp = dr.replace_grad(ray_d, mi.Vector3f(warp))
        return warp, div


    def ray_intersect(self, sdf_shape, sampler, ray, depth=0, ray_test=False, reparam=True, active=True):
        active = mi.Mask(active)
        extra_output = {}

        reparam = reparam and ((self.max_reparam_depth < 0) or (depth <= self.max_reparam_depth))
        with dr.suspend_grad():
            its_result = self.sdf.ray_intersect(dr.detach(ray), warp=self, active=active,
                                                extra_outputs=extra_output if (reparam and self.return_aovs) else None)
            its_t, warp_t, warp_t_d, warp_weight, warp_weight_d = its_result

        div = mi.Float(1.0)
        if reparam:
            warp, div = self.eval(ray(warp_t), ray.d, t=warp_t, dt_dx=warp_t_d,
                                  active=active, extra_output=extra_output,
                                  warp_weight=warp_weight, warp_weight_d=warp_weight_d)
            ray.d = dr.replace_grad(mi.Vector3f(ray.d), warp)
            div = dr.replace_grad(mi.Float(1.0), div)
        if ray_test:
            return dr.isfinite(its_t), div, extra_output
        else:
            si = self.sdf.compute_surface_interaction(ray, its_t)
            si.shape = sdf_shape
            si_d = self.sdf.compute_surface_interaction(dr.detach(ray), dr.detach(its_t))
            si_d.shape = sdf_shape
            return si, si_d, div, extra_output

    def reparam(self, ray, sampler=None, active=True):
        ray = mi.Ray3f(ray)
        det = self.ray_intersect(None, sampler, ray, ray_test=True, reparam=True, active=active)[1]
        return ray.d, det


class WarpFieldConvolution:
    """Normal aligned warp field with analytic divergence computation. Supports distance based downweighting of the warp magnitude."""

    def __init__(self, sdf, n_aux_rays=16):
        self.sdf = sdf
        self.n_aux_rays = n_aux_rays
        self.max_reparam_depth = -1

        self.kappa = 1e5
        self.power = 3.0
        self.return_aovs = False

    def ray_intersect(self, sdf_shape, sampler, ray, depth=0, ray_test=False, reparam=True, active=True):
        from warp_conv import reparameterize_ray, _sample_warp_field
        active = mi.Mask(active)

        reparam = reparam and ((self.max_reparam_depth < 0) or (depth <= self.max_reparam_depth))

        its_t = self.sdf.ray_intersect(dr.detach(ray), active=active)[0]
        div = mi.Float(1.0)
        if reparam:
            # TODO: do we need to pass other parameters here too? not really it seems?
            params = {'sdf.p': self.sdf.p}
            new_d, div = reparameterize_ray(self.sdf, sampler, ray, params=params,
                                            num_auxiliary_rays=self.n_aux_rays, kappa=self.kappa,
                                            power=self.power, active=active)
            ray.d[active] = new_d
            div = dr.replace_grad(mi.Float(1.0), div)
        if ray_test:
            return dr.isfinite(its_t), div, {}
        else:
            si = self.sdf.compute_surface_interaction(ray, its_t)
            si.shape = sdf_shape
            si_d = self.sdf.compute_surface_interaction(dr.detach(ray), dr.detach(its_t))
            si_d.shape = sdf_shape
            return si, si_d, div, {}

    def reparam(self, ray, sampler, active=True):
        from warp_conv import reparameterize_ray
        ray = mi.Ray3f(ray)
        params = {'sdf.p': self.sdf.p}
        new_d, det = reparameterize_ray(self.sdf, sampler, ray, params=params,
                                        num_auxiliary_rays=self.n_aux_rays, kappa=self.kappa,
                                        power=self.power, active=active)
        det = dr.replace_grad(mi.Float(1.0), det)
        return dr.select(active, new_d, ray.d), det


class DummyWarpField:
    def __init__(self, sdf):
        self.sdf = sdf
        self.return_aovs = False

    def ray_intersect(self, sdf_shape, sampler, ray, depth=0, ray_test=False, reparam=True, active=True):
        active = mi.Mask(active)
        extra_output = {}
        its_t = self.sdf.ray_intersect(dr.detach(ray), warp=None, active=active, extra_outputs=None)[0]
        div = mi.Float(1.0)
        if ray_test:
            return dr.isfinite(its_t), div, extra_output
        else:
            si = self.sdf.compute_surface_interaction(ray, its_t)
            si.shape = sdf_shape
            si_d = self.sdf.compute_surface_interaction(dr.detach(ray), dr.detach(its_t))
            si_d.shape = sdf_shape
            return si, si_d, div, extra_output

