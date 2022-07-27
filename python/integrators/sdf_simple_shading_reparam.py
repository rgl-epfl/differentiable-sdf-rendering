import drjit as dr
import mitsuba as mi

from .reparam import ReparamIntegrator


class SdfSimpleShadingReparamIntegrator(ReparamIntegrator):
    """Basic reparameterized SDF integrator that computes a simple, fixed
    shading function. This integrator can be used for debugging or to learn about
    reparameterizations. It is not useful for any actual optimizations."""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_aovs = props.get('use_aovs', False)

    def sample(self, scene, sampler, ray, medium, active):
        reparametrize = True
        si, _, det, extra_output = self.ray_intersect(scene, sampler, ray, reparam=reparametrize)

        result = mi.Spectrum(0.0)
        result[si.is_valid()] = dr.max(dr.dot(si.sh_frame.n, dr.normalize(mi.Vector3f(1, 1, 1))), 0.0)
        result *= det

        aovs = [extra_output[k] if (extra_output is not None) and (k in extra_output)
                else mi.Float(0.0) for k in self.aov_names()]
        return mi.Spectrum(result), mi.Mask(True), det, aovs

    def to_string(self):
        return 'SdfSimpleShadingReparamIntegrator'


mi.register_integrator("sdf_simple_shading_reparam", lambda props: SdfSimpleShadingReparamIntegrator(props))
