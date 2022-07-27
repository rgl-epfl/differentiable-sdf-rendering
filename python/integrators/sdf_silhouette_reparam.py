import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfSilhouetteReparamIntegrator(ReparamIntegrator):
    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)
        self.hide_emitters = False

        props.mark_queried('detach_indirect_si')
        props.mark_queried('decouple_reparam')

    def sample(self, mode, scene, sampler, ray,
               δL,  state_in, reparam, active, **kwargs):

        reparam_primary_ray = True
        si, _, det, extra_output = self.ray_intersect(scene, sampler, ray, reparam=reparam_primary_ray)
        result = dr.select(si.is_valid(), mi.Float(1.0), mi.Float(0.0))
        result *= det

        valid_ray = (not self.hide_emitters) and scene.environment() is not None
        valid_ray |= si.is_valid()

        aovs = [extra_output[k] if (extra_output is not None) and (k in extra_output)
                else mi.Float(0.0) for k in self.aov_names()]
        return mi.Spectrum(result), valid_ray, det, aovs



mi.register_integrator("sdf_silhouette_reparam", lambda props: SdfSilhouetteReparamIntegrator(props))
