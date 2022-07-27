import mitsuba as mi
import drjit as dr


def unit_angle_z(v):
    temp = dr.asin(0.5 * dr.norm(mi.Vector3f(v.x, v.y, v.z - dr.mulsign(mi.Float(1.0), v.z)))) * 2
    return dr.select(v.z >= 0, temp, dr.pi - temp)


class SphericalCamera(mi.Sensor):
    """Defines a spherical sensor that is used for a figure"""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_sphere_uv = props.get('use_sphere_uv', False)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f),
                                                          wavelength_sample, active)
        o = self.world_transform().translation()

        if self.use_sphere_uv:
            d = self.world_transform() @ mi.warp.square_to_uniform_sphere(position_sample)
        else:
            sin_phi, cos_phi = dr.sincos(2 * dr.pi * position_sample.x)
            sin_theta, cos_theta = dr.sincos(dr.pi * position_sample.y)
            d = self.world_transform() @ mi.Vector3f(sin_phi * sin_theta, cos_theta, -cos_phi * sin_theta)
        return mi.Ray3f(o, d, time, wavelengths), wav_weight

    def sample_ray_differential(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        ray, weight = self.sample_ray(time, wavelength_sample, position_sample, aperture_sample, active)
        return mi.RayDifferential3f(ray), weight

    def sample_direction(self, it, sample, active=True):
        # Transform the reference point into the local coordinate system
        trafo = self.world_transform()
        ref_p = trafo.inverse() @ it.p
        d = mi.Vector3f(ref_p)
        dist = dr.norm(d)
        inv_dist = 1.0 / dist
        d *= inv_dist
        resolution = self.film().crop_size()

        ds = dr.zeros(mi.DirectionSample3f)

        if self.use_sphere_uv:
            theta = unit_angle_z(d)
            phi = dr.atan2(d.y, d.x)
            phi[phi < 0.0] += 2 * dr.pi
            ds.uv = mi.Point2f(phi * dr.inv_two_pi, theta * dr.inv_pi)
            ds.uv.x -= dr.floor(ds.uv.x)
            ds.uv *= resolution
            sin_theta = dr.safe_sqrt(1 - d.z * d.z)
        else:
            ds.uv = mi.Point2f(dr.atan2(d.x, -d.z) * dr.inv_two_pi, dr.safe_acos(d.y) * dr.inv_pi)
            ds.uv.x -= dr.floor(ds.uv.x)
            ds.uv *= resolution
            sin_theta = dr.safe_sqrt(1 - d.y * d.y)

        ds.p = trafo.translation()
        ds.d = (ds.p - it.p) * inv_dist
        ds.dist = dist
        ds.pdf = dr.select(active, 1.0, 0.0)

        weight = (1 / (2 * dr.pi * dr.pi * dr.maximum(sin_theta, dr.epsilon(mi.Float)))) * dr.sqr(inv_dist)
        return ds, mi.Spectrum(weight)


mi.register_sensor("spherical", lambda props: SphericalCamera(props))
