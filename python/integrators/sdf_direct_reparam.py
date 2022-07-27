import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import mis_weight

from .reparam import ReparamIntegrator


class SdfDirectReparamIntegrator(ReparamIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.hide_emitters = props.get('hide_emitters', False)
        self.detach_indirect_si = props.get('detach_indirect_si', False)
        self.decouple_reparam = props.get('decouple_reparam', False)

    def sample(self, mode, scene, sampler, ray,
               δL,  state_in, reparam, active, **kwargs):

        active = mi.Mask(active)
        # Reparameterize only if we are not rendering in primal mode
        reparametrize = True and mode != dr.ADMode.Primal
        reparam_primary_ray = True and reparametrize
        si, si_d0, det, extra_output = self.ray_intersect(scene, sampler, ray, depth=0, reparam=reparam_primary_ray)
        valid_ray = (not self.hide_emitters) and scene.environment() is not None
        valid_ray |= si.is_valid()

        throughput = mi.Spectrum(1.0)
        result = mi.Spectrum(0.0)
        throughput *= det
        primary_det = det
        result += throughput * dr.select(active, si.emitter(scene, active).eval(si, active), 0.0)

        ctx = mi.BSDFContext()
        bsdf = si.shape.bsdf()

        # ---------------------- Emitter sampling ----------------------
        active_e = active & si.is_valid() & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        si_d = dr.detach(si)
        with dr.suspend_grad():
            ds, _ = scene.sample_emitter_direction(si_d, sampler.next_2d(active_e), False, active_e)

        active_e &= dr.neq(ds.pdf, 0.0)

        if self.detach_indirect_si:
            shadow_ray = si_d.spawn_ray_to(ds.p)
        elif self.decouple_reparam:
            shadow_ray = si_d0.spawn_ray_to(ds.p)
        else:
            shadow_ray = si.spawn_ray_to(ds.p)

        shadow_ray.d = dr.detach(shadow_ray.d)
        occluded, det_e, extra_output_ = self.ray_test(scene, sampler, shadow_ray, depth=1,
                                                       active=active_e, reparam=reparametrize)
        if not reparam_primary_ray:
            extra_output = extra_output_
        else:
            if self.warp_field is not None:
                if 'warp_t' in extra_output_:
                    extra_output['weight_sum'] = extra_output_['warp_t']

        wo = si.to_local(shadow_ray.d)
        si_e = dr.zeros(mi.SurfaceInteraction3f)
        si_e.sh_frame.n = ds.n
        si_e.initialize_sh_frame()
        si_e.n = si_e.sh_frame.n
        si_e.wi = -shadow_ray.d
        si_e.wavelengths = ray.wavelengths
        emitter_val = dr.select(active_e, ds.emitter.eval(si_e, active_e), 0.0)

        # TODO: Doing this correctly would need recomputing UV differentibably
        # ds.d = shadow_ray.d
        # wo = si.to_local(shadow_ray.d)
        # emitter_val = scene.eval_emitter_direction(dr.detach(si), ds, active_e) # recompute contrib differentiably
        emitter_val = dr.select(ds.pdf > 0, emitter_val / ds.pdf, 0.0)
        visiblity = dr.select(~occluded, 1.0, 0.0)

        if self.use_mis:
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo, active_e)
            nee_contrib = visiblity * bsdf_val * emitter_val * mis_weight(ds.pdf, dr.detach(bsdf_pdf))
        else:
            bsdf_val = bsdf.eval(ctx, si, wo, active_e)
            nee_contrib = visiblity * bsdf_val * emitter_val

        result[active_e] += throughput * nee_contrib * det_e

        # ---------------------- BSDF sampling ----------------------
        if self.use_mis:
            active &= si.is_valid()
            with dr.suspend_grad():
                bs, _ = bsdf.sample(ctx, si_d, sampler.next_1d(active),
                                    sampler.next_2d(active), active)
                active &= bs.pdf > 0.0
            bsdf_ray = si.spawn_ray(si_d.to_world(bs.wo))
            bsdf_ray.d = dr.detach(bsdf_ray.d)
            si_bsdf, si_bsdf_d0, det_bsdf, _ = self.ray_intersect(
                scene, sampler, bsdf_ray, depth=1, reparam=reparametrize)
            bsdf_val = bsdf.eval(ctx, si, bs.wo, active)

            # Determine probability of having sampled that same direction using Emitter sampling.
            with dr.suspend_grad():
                ds = mi.DirectionSample3f(scene, si_bsdf, si)
                delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
                emitter_pdf = dr.select(delta, 0.0, scene.pdf_emitter_direction(si, ds, active))

            emitter_val = si_bsdf.emitter(scene, active).eval(si_bsdf, active)
            bsdf_contrib = bsdf_val / bs.pdf * emitter_val * mis_weight(bs.pdf, emitter_pdf)
            result[active] += throughput * bsdf_contrib * det_bsdf

        aovs = [extra_output[k] if (extra_output is not None) and (k in extra_output)
                else mi.Float(0.0) for k in self.aov_names()]
        return dr.select(valid_ray, mi.Spectrum(result), 0.0), valid_ray, primary_det, aovs


mi.register_integrator("sdf_direct_reparam", lambda props: SdfDirectReparamIntegrator(props))
