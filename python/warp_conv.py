import drjit as dr
import mitsuba as mi

from math_util import normalize_sqr

# Changes compared to reference:
# - replace scene -> sdf everywhere
# - replace si.p - aux_ray.o by our 3D sdf warp field
# - add sdf parameters manually to a dictionary passed to "params"
# - replace boundary test by SDF boundary test

def _sample_warp_field(sdf, sample, ray,
                       kappa, power, active):
    """
    Sample the warp field of moving geometries by tracing an auxiliary ray and
    computing the appropriate convolution weight using the shape's boundary-test.
    """

    # Sample auxiliary direction from vMF
    offset = mi.warp.square_to_von_mises_fisher(sample, kappa)
    omega = mi.Frame3f(dr.detach(ray.d)).to_world(offset)
    pdf_omega = mi.warp.square_to_von_mises_fisher_pdf(offset, kappa)

    aux_ray = mi.Ray3f(ray)
    aux_ray.d = omega

    its_t = sdf.ray_intersect(dr.detach(aux_ray), active=active)[0]
    si = sdf.compute_surface_interaction(aux_ray, its_t)

    hit = active & dr.detach(si.is_valid())

    # Compute warp field direction such that it follows the intersected shape
    # and moves along with the ray origin.

    # SDF 3D motion wrt scene parameters TODO: Here we could potentially improve
    # the gradient quality (for indirect rays) if we somehow compute the
    # dependence of "si.p" on "aux_ray.o" more carefully (but that seems to go
    # beyond whats proposed in the Bangaru et al. paper)
    sdf_value, sdf_normal = sdf.eval_and_grad(dr.detach(si.p))
    sdf_normal_d_n, _ = normalize_sqr(dr.detach(sdf_normal, True))
    warp = -sdf_normal_d_n * sdf_value
    # V_direct = dr.normalize(dr.detach(si.p) + warp - dr.detach(warp) - aux_ray.o)
    V_direct = dr.normalize(dr.detach(si.p) + dr.replace_grad(mi.Vector3f(0.0), warp) - aux_ray.o)

    # Background doesn't move w.r.t. scene parameters
    V_direct = dr.select(hit, V_direct, dr.detach(aux_ray.d))

    # Compute harmonic weight while being careful of division by almost zero
    div_eps = 1e-7

    boundary_test = dr.sqr(dr.dot(si.sh_frame.n, -aux_ray.d))
    B = dr.detach(dr.select(hit, boundary_test, 1.0))
    D = dr.exp(kappa - kappa * dr.dot(ray.d, aux_ray.d)) - 1.0
    w_denom = D + B
    w_denom_p = dr.power(w_denom, power)
    w = dr.select(w_denom > div_eps, dr.rcp(w_denom_p), 0.0)
    w /= pdf_omega
    w = dr.detach(w)

    # Analytic weight gradients w.r.t. `ray.d`
    tmp0 = dr.power(w_denom, power + 1.0)
    tmp1 = (D + 1.0) * dr.select(w_denom > div_eps, dr.rcp(tmp0), 0.0) * kappa * power
    tmp2 = omega - ray.d * dr.dot(ray.d, omega)
    d_w_omega = dr.sign(tmp1) * dr.minimum(dr.abs(tmp1), 1e10) * tmp2
    d_w_omega /= pdf_omega
    d_w_omega = dr.detach(d_w_omega)

    return w, d_w_omega, w * V_direct, dr.dot(d_w_omega, V_direct)


def reparameterize_ray(sdf, sampler, ray, params={}, active=True,
                       num_auxiliary_rays=4, kappa=1e5, power=3.0, use_antithetic_sampling=False):
    """
    Reparameterize given ray by "attaching" the derivatives of its direction to
    moving geometry in the scene.

    Parameter ``scene`` (``mi.Scene``):
        Scene containing all shapes.

    Parameter ``sampler`` (``mi.Sampler``):
        Sampler object used to sample auxiliary rays direction.

    Parameter ``ray`` (``mi.RayDifferential3f``):
        Ray to be reparameterized

    Parameter ``params`` (``mitsuba.python.util.SceneParameters``):
        Scene parameters

    Parameter ``active`` (``mi.Mask``):
        mi.Mask specifying the active lanes

    Parameter ``num_auxiliary_rays`` (``int``):
        Number of auxiliary rays to trace when performing the convolution.

    Parameter ``kappa`` (``float``):
        Kappa parameter of the von Mises Fisher distribution used to sample the
        auxiliary rays.

    Parameter ``power`` (``float``):
        Power value used to control the harmonic weights.

    Returns → (mi.Vector3f, mi.Float):
        Reparameterized ray direction and divergence value of the warp field.
    """

    num_auxiliary_rays = dr.opaque(mi.UInt32, num_auxiliary_rays)
    kappa = dr.opaque(mi.Float, kappa)
    power = dr.opaque(mi.Float, power)

    class Reparameterizer(dr.CustomOp):
        """
        Custom Enoki operator to reparameterize rays in order to account for
        gradient discontinuities due to moving geometry in the scene.
        """
        def eval(self, sdf, params, ray_, active_):
            self.sdf = sdf
            self.ray = ray_
            self.active = active_
            self.add_input(mi.traverse(sdf))
            return self.ray.d, dr.zeros(mi.Float, dr.width(self.ray.d))

        def forward(self):
            loop_record_state = dr.flag(dr.JitFlag.LoopRecord)
            dr.set_flag(dr.JitFlag.LoopRecord, True)

            assert sampler is not None
            use_pcg32 = isinstance(sampler, mi.PCG32)

            # Initialize some accumulators
            Z = mi.Float(0.0)
            dZ = mi.Vector3f(0.0)
            grad_V = mi.Vector3f(0.0)
            grad_div_lhs = mi.Float(0.0)
            ray = mi.Ray3f(dr.detach(self.ray))

            it = mi.UInt32(0)
            loop = mi.Loop(name="reparameterize_ray(): forward propagation",
                        state=lambda: (it, Z, dZ, grad_V, grad_div_lhs, sampler))

            while loop(self.active & (it < num_auxiliary_rays)):

                if use_pcg32:
                    sample = mi.Point2f(sampler.next_float32(),
                                     sampler.next_float32())
                else:
                    sample = sampler.next_2d(self.active)


                dr.enable_grad(ray.o)
                dr.set_grad(ray.o, self.grad_in('ray_').o)
                Z_i, dZ_i, V_i, div_lhs_i = _sample_warp_field(self.sdf, sample, ray,
                                                               kappa, power, self.active)
                dr.enqueue(dr.ADMode.Backward, V_i, div_lhs_i)
                dr.traverse(mi.Float, dr.ADMode.Forward, dr.ADFlag.ClearEdges | dr.ADFlag.ClearInterior)

                Z += Z_i
                dZ += dZ_i
                grad_V += dr.grad(V_i)
                grad_div_lhs += dr.grad(div_lhs_i)
                it = it + 1

                if use_antithetic_sampling:
                    dr.enable_grad(ray.o)
                    dr.set_grad(ray.o, self.grad_in('ray_').o)
                    Z_i, dZ_i, V_i, div_lhs_i = _sample_warp_field(self.sdf, 1 - sample, ray,
                                                                   kappa, power, self.active)
                    dr.enqueue(dr.ADMode.Backward, V_i, div_lhs_i)
                    dr.traverse(mi.Float, dr.ADMode.Forward, dr.ADFlag.ClearEdges | dr.ADFlag.ClearInterior)
                    Z += Z_i
                    dZ += dZ_i
                    grad_V += dr.grad(V_i)
                    grad_div_lhs += dr.grad(div_lhs_i)
                    it = it + 1


            Z = dr.maximum(Z, 1e-8)
            V_theta  = grad_V / Z
            div_V_theta = (grad_div_lhs - dr.dot(V_theta, dZ)) / Z

            # Ignore inactive lanes
            V_theta = dr.select(self.active, V_theta, 0.0)
            div_V_theta = dr.select(self.active, div_V_theta, 0.0)

            self.set_grad_out((V_theta, div_V_theta))

            # Potentially reset loop record flag
            dr.set_flag(dr.JitFlag.LoopRecord, loop_record_state)


        def backward(self):
            loop_record_state = dr.flag(dr.JitFlag.LoopRecord)
            dr.set_flag(dr.JitFlag.LoopRecord, True)
            grad_direction, grad_divergence = self.grad_out()
            use_pcg32 = isinstance(sampler, mi.PCG32)
            assert sampler is not None

            # Ignore inactive lanes
            grad_direction  = dr.select(self.active, grad_direction, 0.0)
            grad_divergence = dr.select(self.active, grad_divergence, 0.0)

            with dr.suspend_grad():
                # We need to trace the auxiliary rays a first time to compute the
                # constants Z and dZ in order to properly weight the incoming gradients
                Z = mi.Float(0.0)
                dZ = mi.Vector3f(0.0)

                if use_pcg32:
                    sampler_clone = mi.PCG32(sampler)
                else:
                    sampler_clone = sampler.clone()

                it = mi.UInt32(0)
                loop = mi.Loop(name="reparameterize_ray(): normalization",
                            state=lambda: (it, Z, dZ, sampler_clone))
                while loop(self.active & (it < num_auxiliary_rays)):
                    if use_pcg32:
                        sample = mi.Point2f(sampler_clone.next_float32(),
                                         sampler_clone.next_float32())
                    else:
                        sample = sampler_clone.next_2d(self.active)

                    Z_i, dZ_i, _, _ = _sample_warp_field(self.sdf, sample, self.ray,
                                                         kappa, power, self.active)
                    Z += Z_i
                    dZ += dZ_i
                    it += 1

                    if use_antithetic_sampling:
                        Z_i, dZ_i, _, _ = _sample_warp_field(self.sdf, 1 - sample, self.ray,
                                                             kappa, power, self.active)
                        Z += Z_i
                        dZ += dZ_i
                        it += 1


            # Un-normalized values
            V = dr.zeros(mi.Vector3f, dr.width(Z))
            div_V_1 = dr.zeros(mi.Float, dr.width(Z))
            dr.enable_grad(V, div_V_1)

            # Compute normalized values
            Z = dr.maximum(Z, 1e-8)
            V_theta = V / Z
            divergence = (div_V_1 - dr.dot(V_theta, dZ)) / Z
            direction = dr.normalize(self.ray.d + V_theta)

            dr.set_grad(direction, grad_direction)
            dr.set_grad(divergence, grad_divergence)
            dr.enqueue(dr.ADMode.Backward, direction, divergence)
            dr.traverse(mi.Float, dr.ADMode.Backward)

            grad_V = dr.grad(V)
            grad_div_V_1 = dr.grad(div_V_1)

            it = mi.UInt32(0)
            loop = mi.Loop(name="reparameterize_ray(): backpropagation",
                        state=lambda: (it, sampler))
            while loop(self.active & (it < num_auxiliary_rays)):
                if use_pcg32:
                    sample = mi.Point2f(sampler.next_float32(),
                                     sampler.next_float32())
                else:
                    sample = sampler.next_2d(self.active)

                _, _, V_i, div_V_1_i = _sample_warp_field(self.sdf, sample, self.ray,
                                                          kappa, power, self.active)
                dr.set_grad(V_i, grad_V)
                dr.set_grad(div_V_1_i, grad_div_V_1)
                dr.enqueue(dr.ADMode.Backward, V_i, div_V_1_i)
                dr.traverse(mi.Float, dr.ADMode.Backward, dr.ADFlag.ClearVertices)
                it += 1

                if use_antithetic_sampling:
                    _, _, V_i, div_V_1_i = _sample_warp_field(self.sdf, 1 - sample, self.ray,
                                                              kappa, power, self.active)
                    dr.set_grad(V_i, grad_V)
                    dr.set_grad(div_V_1_i, grad_div_V_1)
                    dr.enqueue(dr.ADMode.Backward, V_i, div_V_1_i)
                    dr.traverse(mi.Float, dr.ADMode.Backward, dr.ADFlag.ClearVertices)
                    it += 1

            # Potentially reset loop record flag
            dr.set_flag(dr.JitFlag.LoopRecord, loop_record_state)

        def name(self):
            return "ray reparameterizer"

    return dr.custom(Reparameterizer, sdf, params, ray, active)
