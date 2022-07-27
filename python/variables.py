"""Contains logic to initialize, update and read/write optimized variables.

The code base currently only supports volumetric variables, but the interface
could also be used for other types of variables.
"""

import os

import drjit as dr
import mitsuba as mi
import numpy as np

from shapes import BoxSDF, Grid3d, create_sphere_sdf

import redistancing
from util import atleast_4d

def upsample_sdf(sdf_data):
    new_res = 2 * mi.ScalarVector3i(sdf_data.shape[:3])
    sdf = Grid3d(sdf_data)
    z, y, x = dr.meshgrid(*[(dr.arange(mi.Int32, new_res[i]) + 0.5) / new_res[i] for i in range(3)], indexing='ij')
    sdf = type(sdf_data)(dr.detach(sdf.eval(mi.Point3f(x, y, z))), new_res)
    return atleast_4d(sdf)

def upsample_grid(data):
    return dr.upsample(mi.Texture3f(data, migrate=False), scale_factor=[2, 2, 2, 1]).tensor()

def simple_lr_decay(initial_lrate, decay, i):
    lr = initial_lrate / (1 + decay * i)

    # Hardcoded for now: further decay LR as target (512 iterations) is reached
    if i > 480:
        lr = lr / 2
    if i > 500:
        lr = lr / 2
    return lr


class Variable:
    """Represents a variable in an optimization that can be initialized, updated and saved"""

    def __init__(self, k, beta=None, regularizer_weight=0.0, regularizer=None, lr=None):
        self.k = k
        self.mean = None
        self.beta = beta
        self.regularizer_weight = regularizer_weight
        self.regularizer = regularizer
        self.lr = None

    def initialize(self, opt):
        return

    def save(self, opt, output_dir, suffix):
        return

    def restore(self, opt, output_dir, suffix):
        return

    def validate_gradient(self, opt, i):
        return

    def validate(self, opt, i):
        return

    def update_mean(self, opt, i):
        return

    def load_mean(self, opt):
        if self.mean is not None:
            opt[self.k] = self.mean

    def eval_regularizer(self, opt, sdf_object, i):
        if self.regularizer is not None:
            return self.regularizer_weight * self.regularizer(opt[self.k])
        else:
            return 0.0


class VolumeVariable(Variable):
    def __init__(self, k, shape, init_value=0.5, upsample_iter=[64, 128], **kwargs):
        super().__init__(k, **kwargs)
        self.shape = np.array(shape)
        self.init_value = init_value
        self.upsample_iter = upsample_iter
        if self.upsample_iter is not None:
            self.upsample_iter = list(self.upsample_iter)
            for i in range(3):
                self.shape[i] = self.shape[i] // 2 ** len(self.upsample_iter)

    def initialize(self, opt):
        opt[self.k] = dr.full(mi.TensorXf, self.init_value, self.shape)

        if self.lr is not None:
            opt.set_learning_rate({self.k, self.lr})

    def get_variable_path(self, output_dir, suffix, suffix2=''):
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{self.k.replace(".", "-")}-{suffix_str}{suffix2}.vol')

    def save(self, opt, output_dir, suffix):
        mi.VolumeGrid(np.array(opt[self.k])).write(self.get_variable_path(output_dir, suffix))

    def restore(self, opt, output_dir, suffix):
        loaded_data = np.array(mi.VolumeGrid(self.get_variable_path(output_dir, suffix)))
        # Make sure the number of dimensions matches up
        if self.k in opt and (opt[self.k].ndim == 4 and loaded_data.ndim == 3):
            loaded_data = loaded_data[..., None]
        opt[self.k] = mi.TensorXf(loaded_data)

    def validate(self, opt, i):
        k = self.k
        if self.upsample_iter is not None and i in self.upsample_iter:
            opt[k] = mi.TensorXf(upsample_grid(opt[k]))

        if k.endswith('reflectance.volume.data') or k.endswith('base_color.volume.data'):
            opt[k] = dr.clamp(opt[k], 1e-5, 1.0)
        if k.endswith('roughness.volume.data'):
            opt[k] = dr.clamp(opt[k], 0.1, 0.8)
        dr.enable_grad(opt[k])

    def update_mean(self, opt, i):
        if self.beta is None:
            return

        if self.mean is None or (opt[self.k].shape != self.mean.shape):
            self.mean = dr.detach(mi.TensorXf(opt[self.k]), True)
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * dr.detach(opt[self.k], True)

        # This is crucial to prevent enoki from building a
        # graph across iterations, leading to growth in memory use
        dr.schedule(self.mean)


class SdfVariable(VolumeVariable):

    def __init__(self, k, resolution,
                 sdf_init_fn=create_sphere_sdf,
                 adaptive_learning_rate=True, **kwargs):
        super().__init__(k, shape=(resolution,) * 3, **kwargs)
        self.adaptive_learning_rate = adaptive_learning_rate
        self.bbox_constraint = True
        self.sdf_init_fn = sdf_init_fn
        if self.bbox_constraint:
            self.update_box_sdf(self.shape)
        self.lr_decay_rate = 0.02

    def initialize(self, opt):
        self.initial_lr = opt.lr[self.k]
        self.initial_shape = self.shape
        opt[self.k] = atleast_4d(mi.TensorXf(self.sdf_init_fn(self.shape)))
        if self.lr is not None:
            opt.set_learning_rate({self.k, self.lr})

    # Overload variable path to not use "SamplingIntegrator" as a prefix
    def get_variable_path(self, output_dir, suffix, suffix2=''):
        k = self.k.replace("SamplingIntegrator.", "")
        suffix_str = f'{suffix:04d}' if isinstance(suffix, int) else suffix
        return os.path.join(output_dir, f'{k.replace(".", "-")}-{suffix_str}{suffix2}.vol')

    def update_box_sdf(self, res):
        bbox = BoxSDF(mi.Point3f(0), mi.Vector3f(0.49), smoothing=0.01)
        z, y, x = dr.meshgrid(dr.linspace(mi.Float, -0.5, 0.5, res[0]),
                              dr.linspace(mi.Float, -0.5, 0.5, res[1]),
                              dr.linspace(mi.Float, -0.5, 0.5, res[2]), indexing='ij')
        self.bbox_sdf = atleast_4d(mi.TensorXf(bbox.eval(mi.Point3f(x, y, z)), res))

    def validate(self, opt, i):
        k = self.k
        if self.upsample_iter is not None and i in self.upsample_iter:
            sdf = upsample_sdf(opt[k])
            self.shape = sdf.shape
            if self.bbox_constraint:
                self.update_box_sdf(self.shape)
        else:
            self.shape = opt[k].shape
            sdf = opt[k]

        if self.adaptive_learning_rate and i is not None:
            # Scale LR according to SDF res compared to a "32x32x32" "reference" case
            lr_scale = 32 / self.shape[0]
            lr = lr_scale * simple_lr_decay(self.initial_lr, self.lr_decay_rate, i)
            opt.set_learning_rate({k: lr})

        if self.bbox_constraint:
            assert sdf.shape == self.bbox_sdf.shape
            sdf = dr.maximum(sdf, self.bbox_sdf)

        sdf = redistancing.redistance(sdf)
        opt[k] = atleast_4d(mi.TensorXf(sdf))
        dr.enable_grad(opt[k])

    def validate_gradient(self, opt, i):
        k = self.k
        grad = dr.grad(opt[k])

        # Clamp gradients and suppress NaNs just in case
        r = 1e-1
        dr.set_grad(opt[k], dr.select(dr.isnan(grad), 0.0, dr.clamp(grad, -r, r)))

    def eval_regularizer(self, opt, sdf_object, i):
        if self.regularizer is not None and self.regularizer_weight > 0.0:
            return self.regularizer_weight * self.regularizer(opt[self.k], sdf_object)
        else:
            return 0.0
