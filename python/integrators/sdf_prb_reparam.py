from __future__ import annotations

import gc

import drjit as dr
import mitsuba as mi
import warp
from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator


class ReparamWrapper:
    DRJIT_STRUCT = {'rng': mi.PCG32}

    def __init__(self, params, reparam, wavefront_size, seed):
        self.params = params
        self.reparam = reparam
        # # Only link the reparameterization CustomOp to differentiable shape params
        # if isinstance(params, mi.util.SceneParameters):
        #     params = params.copy()
        #     params.keep_shape()
        idx = dr.arange(mi.UInt32, wavefront_size)
        tmp = dr.opaque(mi.UInt32, 0xffffffff ^ seed)
        v0, v1 = mi.sample_tea_32(tmp, idx)
        self.rng = mi.PCG32(initstate=v0, initseq=v1)

    def __call__(self, ray, depth, active=True):
        return self.reparam(ray, self.rng,
                            depth, active)


class SdfPrbReparamIntegrator(ReparamIntegrator):

    def __init__(self, props):
        super().__init__(props)
        self.reparam_max_depth = props.get('reparam_max_depth', self.max_depth)
        self.shade_gradient_max_depth = self.reparam_max_depth
        self.use_aovs = props.get('use_aovs', False)
        self.hide_emitters = props.get('hide_emitters', False)
        self.rr_depth = props.get('rr_depth', 5)

        self.reparam_rays = props.get('reparam_rays', 16)
        self.reparam_kappa = props.get('reparam_kappa', 1e5)
        self.reparam_exp = props.get('reparam_exp', 3.0)

        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')

    def aovs(self):
        return []

    # Copy and adapat from `common.py`
    def render(self, scene, sensor=0, seed=0, spp=0,
               develop=True, evaluate=True):

        self.prepare_sdf(scene)
        if not develop:
            raise Exception("develop=True must be specified when invoking AD integrators")
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            sampler, spp = self.prepare(sensor=sensor, seed=seed, spp=spp, aovs=self.aovs())
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)
            L, valid, state = self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler, ray=ray,
                depth=mi.UInt32(0), δL=None, state_in=None, reparam=None, active=mi.Bool(True))
            block = sensor.film().create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))
            block.put(pos, ray.wavelengths, L * weight, alpha)
            del sampler, ray, weight, pos, L, valid, state, alpha
            gc.collect()
            sensor.film().put_block(block)
            self.primal_image = sensor.film().develop()
            return self.primal_image


    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:

        self.prepare_sdf(scene)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Pass reparametrize function, if necessary use the reparam wrapper to guarantee decorrelated random samples
            if self.warp_field is not None:
                if isinstance(self.warp_field, warp.WarpFieldConvolution):
                    def reparam(ray, sampler, depth, active=True):
                        assert sampler is not None
                        return self.warp_field.reparam(ray, sampler=sampler, active=active & (depth <= self.reparam_max_depth))
                    reparam = ReparamWrapper(params=params, reparam=reparam,
                                             wavefront_size=sampler.wavefront_size(), seed=seed)
                else:
                    def reparam(ray, depth, active=True):
                        return self.warp_field.reparam(ray, active=active & (depth <= self.reparam_max_depth))
            else:
                reparam = None
            ray, weight, pos, primary_det = self.sample_rays(scene, sensor, sampler, reparam)
            L, valid, state_out = self.sample(mode=dr.ADMode.Primal, scene=scene, sampler=sampler.clone(),
                                              ray=ray, depth=mi.UInt32(0), δL=None, state_in=None, reparam=None, active=mi.Bool(True))
            sample_pos_deriv = None # disable by default
            with dr.resume_grad():
                if dr.grad_enabled(pos, primary_det):
                    sample_pos_deriv = film.create_block()
                    sample_pos_deriv.set_coalesce(sample_pos_deriv.coalesce() and spp >= 4)
                    sample_pos_deriv.put(pos=pos, wavelengths=ray.wavelengths, value=L * weight * primary_det,
                                         alpha=dr.select(valid, mi.Float(1), mi.Float(0)), weight=primary_det)
                    tensor = sample_pos_deriv.tensor()
                    dr.forward_to(tensor, flags=dr.ADFlag.ClearInterior | dr.ADFlag.ClearEdges)
                    dr.schedule(tensor, dr.grad(tensor))
                    dr.disable_grad(pos)
                    del tensor

            gc.collect()
            dr.eval(state_out)
            del L, valid, params
            δL, valid_2, state_out_2 = self.sample(mode=dr.ADMode.Forward, scene=scene, sampler=sampler, ray=ray,
                                                   depth=mi.UInt32(0),  δL=None, state_in=state_out, reparam=reparam, active=mi.Bool(True))
            block = film.create_block()
            block.set_coalesce(block.coalesce() and spp >= 4)
            block.put(pos=pos,
                      wavelengths=ray.wavelengths,
                      value=δL * weight,
                      alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)))
            film.put_block(block)
            del sampler, ray, weight, pos, δL, valid_2, state_out, state_out_2, block
            gc.collect()
            result_grad = film.develop()
            if sample_pos_deriv is not None:
                with dr.resume_grad():
                    film.prepare(aovs)
                    film.put_block(sample_pos_deriv)
                    reparam_result = film.develop()
                    dr.forward_to(reparam_result)
                    result_grad += dr.grad(reparam_result)
        return result_grad

    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: int = 0,
                        spp: int = 0) -> None:

        self.prepare_sdf(scene)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        aovs = self.aovs()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, aovs)

            # Pass reparametrize function, if necessary use the reparam wrapper to guarantee decorrelated random samples
            if self.warp_field is not None:
                if isinstance(self.warp_field, warp.WarpFieldConvolution):
                    def reparam(ray, sampler, depth, active=True):
                        assert sampler is not None
                        return self.warp_field.reparam(ray, sampler=sampler, active=active & (depth <= self.reparam_max_depth))
                    reparam = ReparamWrapper(params=params, reparam=reparam,
                                             wavefront_size=sampler.wavefront_size(), seed=seed)
                else:
                    def reparam(ray, depth, active=True):
                        return self.warp_field.reparam(ray, active=active & (depth <= self.reparam_max_depth))
            else:
                reparam = None

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos, det = self.sample_rays(scene, sensor,
                                                     sampler, reparam)

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, state_out = self.sample(
                mode=dr.ADMode.Primal, scene=scene, sampler=sampler.clone(),
                ray=ray, depth=mi.UInt32(0), δL=None, state_in=None, reparam=None, active=mi.Bool(True))

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            with dr.resume_grad():
                dr.enable_grad(L)
                block.put(pos=pos, wavelengths=ray.wavelengths,
                          value=L * weight * det, weight=det, alpha=dr.select(valid, mi.Float(1), mi.Float(0)))

                sensor.film().put_block(block)
                del valid
                gc.collect()

                dr.schedule(state_out, block.tensor())
                image = sensor.film().develop()

                # Differentiate sample splatting and weight division
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(mi.Float, dr.ADMode.Backward)
                δL = dr.grad(L)

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, state_out_2 = self.sample(
                mode=dr.ADMode.Backward, scene=scene, sampler=sampler,
                ray=ray, depth=mi.UInt32(0), δL=δL,
                state_in=state_out, reparam=reparam, active=mi.Bool(True))

            # We don't need any of the outputs here
            del L_2, valid_2, state_out, state_out_2, δL, \
                ray, weight, pos, block, sampler

            gc.collect()

            # Run kernel representing side effects of the above
            dr.eval()

    def sample(self, mode: enoki.ADMode,  scene: mi.Scene,
               sampler: mi.Sampler, ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],  state_in: Optional[mi.Spectrum],
               reparam: Optional[Callable[[mi.Ray3f, mi.Bool], Tuple[mi.Ray3f, mi.Float]]],
               active: mi.Bool, **kwargs):

        primal = mode == dr.ADMode.Primal
        bsdf_ctx = mi.BSDFContext()

        # If no reparam is specified, just pass through
        if reparam is None:
            reparam = lambda r, d, active_=True: (r.d, mi.Float(1.0))

        valid_ray = (not self.hide_emitters) and scene.environment() is not None

        # --------------------- Configure loop state ----------------------
        depth = mi.UInt32(0)
        L = mi.Spectrum(0 if primal else state_in)
        δL = mi.Spectrum(δL if δL is not None else 0)
        β = mi.Spectrum(1)
        η = mi.Float(1)
        mis_em = mi.Float(1)
        active = mi.Bool(active)

        ray_prev = dr.zeros(mi.Ray3f)
        ray_cur = mi.Ray3f(ray)
        pi_prev = dr.zeros(mi.PreliminaryIntersection3f)
        pi_cur = self.ray_intersect_preliminary(scene, ray_cur, active=active)
        valid_ray |= pi_cur.is_valid()

        if self.warp_field is not None:
            m = self.warp_field.max_reparam_depth
            self.reparam_max_depth = m if m >= 0 else 100000
            # If we limit the reparam depth, also limit the shading gradient depth
            # to fully disable indirect gradients
            self.shade_gradient_max_depth = self.reparam_max_depth

        # Record the following loop in its entirety
        if hasattr(reparam, 'DRJIT_STRUCT'):
            loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                        state=lambda: (sampler, depth, L, δL, β, η, mis_em, active,
                                       ray_prev, ray_cur, pi_prev, pi_cur, reparam))
        else:
            loop = mi.Loop(name="Path Replay Backpropagation (%s)" % mode.name,
                        state=lambda: (sampler, depth, L, δL, β, η, mis_em, active,
                                       ray_prev, ray_cur, pi_prev, pi_cur))
        loop.set_max_iterations(self.max_depth)
        while loop(active):
            first_vertex = dr.eq(depth, 0)
            ray_reparam = mi.Ray3f(ray_cur)
            ray_reparam_det = 1

            # ----------- Reparameterize (differential phase only) -----------
            if not primal:
                with dr.resume_grad():
                    # Differentiably recompute prev. interaction and reparametrize the current ray, that is starting from that one
                    si_prev = self.compute_surface_interaction(pi_prev, ray_prev, mi.RayFlags.All | mi.RayFlags.FollowShape)
                    ray_reparam.d, ray_reparam_det = reparam(dr.select(first_vertex, ray_cur, si_prev.spawn_ray(ray_cur.d)), depth)
                    ray_reparam_det[first_vertex] = 1
                    dr.disable_grad(si_prev)

            with dr.resume_grad(when=not primal):
                #  Compute differentiable record of the current interaction
                si_cur = self.compute_surface_interaction(pi_cur, ray_reparam)

                # Differentiably evaluate direct emission
                emitter = si_cur.emitter(scene)
                Le = β * mis_em * emitter.eval(si_cur) # MIS with NEE from prev. bounce

            # Next event estimation
            active_next = (depth + 1 < self.max_depth) & si_cur.is_valid()
            bsdf_cur = si_cur.bsdf(ray_cur)
            active_em = active_next & mi.has_flag(bsdf_cur.flags(), mi.BSDFFlags.Smooth)
            ds, em_weight = scene.sample_emitter_direction(si_cur, sampler.next_2d(), False, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            with dr.resume_grad(when=not primal):
                em_ray_det = 1
                em_ray = si_cur.spawn_ray_to(ds.p)
                em_ray.d = dr.detach(em_ray.d)

                # Reparametrized ray test
                # TODO: The depth argument here is currently "static" to just support disabling higher order grads
                occluded, em_ray_det, _ = self.ray_test(scene, sampler, em_ray, depth=1,
                                                        reparam=not primal, active=active_em)
                active_em &= ~occluded
                if not primal:
                    ds.d = em_ray.d
                    em_val = scene.eval_emitter_direction(dr.detach(si_cur), ds, active_em) # recompute contrib differentiably
                    em_weight = dr.select(dr.neq(ds.pdf, 0) & ~occluded, em_val / ds.pdf, 0)

                # Evaluate BSDF etc using the reparametrized ray
                wo = si_cur.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf_cur.eval_pdf(bsdf_ctx, si_cur, wo, active_em)
                mis_direct = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
                Lr_dir = β * dr.detach(mis_direct) * bsdf_value_em * em_weight * em_ray_det

            # BSDF Sampling
            bsdf_sample, bsdf_weight = bsdf_cur.sample(bsdf_ctx, si_cur, sampler.next_1d(), sampler.next_2d(), active_next)
            η *= bsdf_sample.eta
            β *= bsdf_weight
            L_prev = L # Value of 'L' at previous vertex
            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir) # PRB update rule

            # Stopping criterion
            β_max = dr.max(β)
            active_next &= dr.neq(β_max, 0)
            rr_prob = dr.minimum(β_max * η**2, .95)
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # Intersect next surface-
            ray_next = si_cur.spawn_ray(si_cur.to_world(bsdf_sample.wo))
            pi_next = self.ray_intersect_preliminary(scene, ray_next, active=active_next)
            si_next = self.compute_surface_interaction(pi_next, ray_next)

            # Compute MIS weight for the next vertex
            # (using probability of sampling 'si_next' using emitter sampling)

            ds = mi.DirectionSample3f(scene, si=si_next, ref=si_cur)
            bsdf_sample_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            pdf_em = scene.pdf_emitter_direction(ref=si_cur, ds=ds, active=~bsdf_sample_delta)
            mis_em = mis_weight(bsdf_sample.pdf, pdf_em)

            if not primal:
                sampler_clone = sampler.clone()
                active_next_next = active_next & si_next.is_valid() & (depth + 2 < self.max_depth)

                # Retrieve the BSDFs of the two adjacent vertices
                bsdf_next = si_next.bsdf(ray_next)
                bsdf_prev = si_prev.bsdf(ray_prev)

                active_em_next = active_next_next & mi.has_flag(bsdf_next.flags(), mi.BSDFFlags.Smooth)
                ds_next, em_weight_next = scene.sample_emitter_direction(si_next, sampler_clone.next_2d(), True, active_em_next)
                active_em_next &= dr.neq(ds_next.pdf, 0.0)

                # Compute the emission sampling contribution at the next vertex
                bsdf_value_em_next, bsdf_pdf_em_next = bsdf_next.eval_pdf(bsdf_ctx, si_next, si_next.to_local(ds_next.d), active_em_next)
                mis_direct_next = dr.select(ds_next.delta, 1, mis_weight(ds_next.pdf, bsdf_pdf_em_next))
                Lr_dir_next = β * mis_direct_next * bsdf_value_em_next * em_weight_next

                # Generate a detached BSDF sample at the next vertex
                bsdf_sample_next, _ = bsdf_next.sample(bsdf_ctx, si_next, sampler_clone.next_1d(), sampler_clone.next_2d(), active_next_next)

                # Account for adjacent vertices, but only consider derivatives
                # that arise from the reparameterization at 'si_cur.p'
                with dr.resume_grad(si_cur.p): # TODO: Double check these `si_cur` terms
                    # Differentiably recompute the outgoing direction at 'prev'  and the incident direction at 'next'
                    wo_prev = dr.normalize(si_cur.p - si_prev.p)
                    wi_next = dr.normalize(si_cur.p - si_next.p)

                    # Compute the emission at the next vertex to compute L_next
                    si_next.wi = si_next.to_local(wi_next) # make incident direction differentiable here
                    Le_next = β * mis_em * si_next.emitter(scene).eval(si_next, active_next)
                    L_next = L - dr.detach(Le_next) - dr.detach(Lr_dir_next)

                    # The previous BSDF eval derivative due to reparametrizing
                    bsdf_val_prev = bsdf_prev.eval(bsdf_ctx, si_prev, si_prev.to_local(wo_prev))

                    # TODO: What's the role of this term really?
                    bsdf_val_next = bsdf_next.eval(bsdf_ctx, si_next, bsdf_sample_next.wo)

                    # terms related to previous and next vertex' BSDF evaluation
                    extra = mi.Spectrum(Le_next)
                    extra[~first_vertex] += L_prev * bsdf_val_prev / dr.detach(bsdf_val_prev)
                    extra[si_next.is_valid()] += L_next * bsdf_val_next / dr.detach(bsdf_val_next)
                with dr.resume_grad():
                    wo = si_cur.to_local(ray_next.d) # allow cosine term derivs
                    bsdf_val = bsdf_cur.eval(bsdf_ctx, si_cur, wo, active_next)
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(dr.neq(bsdf_val_det, 0), dr.rcp(bsdf_val_det), 0)
                    Lr_ind = L * dr.replace_grad(1, inv_bsdf_val_det * bsdf_val)

                    Lo = (Le + Lr_dir + Lr_ind) * ray_reparam_det + extra

                    Lo[depth > self.shade_gradient_max_depth] = 0

                    if dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo):
                        raise Exception("No gradients enabled!")
                    if mode == dr.ADMode.Backward:
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            if not primal:
                pi_prev = pi_cur
                ray_prev = ray_cur
            pi_cur = pi_next
            ray_cur = ray_next
            depth[si_cur.is_valid()] += 1
            active = active_next

        return (L if primal else δL), valid_ray, L

    def sample_rays(self, scene, sensor, sampler, reparam=None):
        """
        Sample a 2D grid of primary rays for a given sensor

        Returns a tuple containing

        - the set of sampled rays
        - a ray weight (usually 1 if the sensor's response function is sampled
          perfectly)
        - the continuous 2D image-space positions associated with each ray

        When a reparameterization function is provided via the 'reparam'
        argument, it will be applied to the returned image-space position (i.e.
        the sample positions will be moving). The other two return values
        remain detached.
        """

        film = sensor.film()
        film_size = film.crop_size()
        rfilter = film.rfilter()
        border_size = rfilter.border_size()

        if film.sample_border():
            film_size += 2 * border_size

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2u()
        pos.y = idx // film_size[0]
        pos.x = dr.fma(-film_size[0], pos.y, idx)

        if film.sample_border():
            pos -= border_size

        pos += film.crop_offset()

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
        offset = -mi.ScalarVector2f(film.crop_offset()) * scale
        pos_adjusted = dr.fma(pos_f, scale, offset)

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        ray, weight = sensor.sample_ray_differential(time=wavelength_sample, sample1=sampler.next_1d(),
                                                     sample2=pos_adjusted,sample3=aperture_sample)
        det = mi.Float(1.0)
        if reparam is not None:
            assert not rfilter.is_box_filter()
            assert film.sample_border()

            with dr.resume_grad():
                reparam_d, det = reparam(ray=ray, depth=mi.UInt32(0))

                # Create a fake interaction along the sampled ray and use it to the
                # position with derivative tracking
                it = dr.zeros(mi.Interaction3f)
                it.p = ray.o + reparam_d
                ds, _ = sensor.sample_direction(it, aperture_sample)
                # Return a reparameterized image position
                pos_f = ds.uv + film.crop_offset()

        return ray, weight, pos_f, det


mi.register_integrator("sdf_prb_reparam", lambda props: SdfPrbReparamIntegrator(props))
