"""This file contains a list of "optimization configurations" that can be used to
optimize a scene. Configurations are dictionaries and support "inheritance" from
parent configurations.
"""

import os

import numpy as np
import mitsuba as mi

import losses
from shapes import create_sphere_sdf
from variables import VolumeVariable, SdfVariable
from util import get_file_sensors, get_regular_cameras, set_sensor_res, get_regular_cameras_top

import regularizations as reg
from configs import apply_cmdline_args

from constants import SDF_DEFAULT_KEY


class SceneConfig:
    def __init__(self, name, param_keys, sensors=[0, 1, 2], pretty_name=None,
                 resx=64, resy=64, batch_size=None, reorder_sensors=True, param_averaging_beta=0.5):
        self.name = name
        self.sensors = sensors
        if callable(self.sensors):
            self.sensors = self.sensors()
        self.pretty_name = pretty_name if pretty_name else name.capitalize()
        self.loss = losses.l1
        self.resx = resx
        self.resy = resy
        self.target_res = mi.ScalarPoint2i(resy, resx)
        self.init_res = self.target_res
        self.param_keys = param_keys
        self.checkpoint_frequency = 64
        self.variables = []
        self.batch_size = batch_size if batch_size is not None else len(sensors)
        self.param_averaging_beta = param_averaging_beta

    def eval_regularizer(self, opt, sdf_object, i):
        return 0.0

    def save_params(self, opt, output_dir, i, force=False):
        return

    def validate_gradients(self, opt, i):
        return

    def validate_params(self, opt, i):
        return

    def update_scene(self, scene, i):
        """Perform any additional changes to the scene (eg. sensors)"""
        pass

    def get_sensor_iterator(self, i):
        n_sensors = len(self.sensors)
        if self.batch_size and (self.batch_size < n_sensors):
            # Access sensors in a strided way, assuming that this will maximize angular coverage per iteration
            steps = int(np.ceil(n_sensors / self.batch_size))
            indices = [(j * steps + i % steps) % n_sensors for j in range(self.batch_size)]
            sensors = [self.sensors[idx] for idx in indices]
            return zip(indices, sensors)
        else:
            return enumerate(self.sensors)

    def load_checkpoint(self, scene, output_dir, i):
        """Attempts to restore scene parameters given an iteration"""
        params = mi.traverse(scene)
        params.keep(self.param_keys + [SDF_DEFAULT_KEY, ])

        # Create an optimizer that temporarily serves to hold the loaded variables
        opt = mi.ad.Adam(lr=0.1, params=params)
        param_dir = os.path.join(output_dir, 'params')
        for v in self.variables:
            v.restore(opt, param_dir, i)

        params.update(opt)


class SdfConfig(SceneConfig):
    def __init__(self, name, param_keys=[SDF_DEFAULT_KEY],
                 sensors=[0, 1, 2],
                 pretty_name=None,
                 sdf_res=64,
                 sdf_init_fn=create_sphere_sdf,
                 resx=64, resy=64,
                 upsample_iter=[64, 128],
                 loss=losses.l1,
                 use_multiscale_rendering=False,
                 render_upsample_iter=[64, 128],
                 sdf_regularizer_weight=0.0,
                 sdf_regularizer=None,
                 batch_size=None,
                 adaptive_learning_rate=True,
                 tex_upsample_iter=[100, 128, 160, 170, 192],
                 reorder_sensors=True,
                 texture_lr=None,
                 param_averaging_beta=0.1,
                 tex_init_value=0.5):

        super().__init__(name, param_keys=param_keys, sensors=sensors, pretty_name=pretty_name, resx=resx, resy=resy,
                         batch_size=batch_size, reorder_sensors=reorder_sensors,
                         param_averaging_beta=param_averaging_beta)

        sdf = SdfVariable(SDF_DEFAULT_KEY, sdf_res, upsample_iter=upsample_iter,
                          sdf_init_fn=sdf_init_fn, adaptive_learning_rate=adaptive_learning_rate, beta=self.param_averaging_beta,
                          regularizer=sdf_regularizer, regularizer_weight=sdf_regularizer_weight)

        self.variables.append(sdf)
        if len(param_keys) > 1 and ('reflectance' in param_keys[1] or 'base_color' in param_keys[1]):
            self.variables.append(VolumeVariable(param_keys[1], (sdf_res, sdf_res, sdf_res, 3),
                                                init_value=tex_init_value,
                                                 upsample_iter=tex_upsample_iter, beta=self.param_averaging_beta, lr=texture_lr))

        if len(param_keys) > 2 and 'roughness' in param_keys[2]:
            self.variables.append(VolumeVariable(param_keys[2], (sdf_res // 4, sdf_res // 4, sdf_res // 4, 1),
                                                 upsample_iter=[128, 180], beta=self.param_averaging_beta, lr=texture_lr))

        self.loss = loss
        self.render_upsample_iter = None
        if use_multiscale_rendering:
            self.render_upsample_iter = list(render_upsample_iter)
            self.init_res = mi.ScalarPoint2i(np.array(self.target_res) // 2 ** len(self.render_upsample_iter))
        else:
            self.init_res = mi.ScalarPoint2i(self.resx, self.resy)

    def initialize(self, opt, scene):
        if callable(self.sensors):
            self.sensors = self.sensors()

        for v in self.variables:
            v.initialize(opt)
        for sensor in self.sensors:
            set_sensor_res(sensor, self.init_res)

    def validate_params(self, opt, i):
        for v in self.variables:
            v.validate(opt, i)
            v.update_mean(opt, i)

    def load_mean_parameters(self, opt):
        for v in self.variables:
            v.load_mean(opt)
            v.validate(opt, i=None)

    def validate_gradients(self, opt, i):
        for v in self.variables:
            v.validate_gradient(opt, i)

    def save_params(self, opt, output_dir, i, force=False):
        if isinstance(i, str) or (i % self.checkpoint_frequency == 0) or force:
            param_dir = os.path.join(output_dir, 'params')
            os.makedirs(param_dir, exist_ok=True)
            for v in self.variables:
                v.save(opt, param_dir, i)

    def update_scene(self, scene, i):
        # Double the sensor resolution as needed
        if self.render_upsample_iter is not None and i in self.render_upsample_iter:
            target_res = self.init_res * 2 ** (sorted(self.render_upsample_iter).index(i) + 1)
            for sensor in self.sensors:
                set_sensor_res(sensor, target_res)

    def eval_regularizer(self, opt, sdf_object, i):
        reg = 0.0
        for v in self.variables:
            reg += v.eval_regularizer(opt, sdf_object, i)
        return reg


SCENE_CONFIGS = {}


def create_scene_config_init_fn(name, config_class, sensors, scene_name=None, resx=128, resy=128, **kwargs):
    if not scene_name:
        scene_name = name

    # Support passing a list of sensors: then just use the default sensors from the scene
    if sensors is None or (isinstance(sensors, list) and isinstance(sensors[0], int)):
        sensors_list = lambda: get_file_sensors(os.path.join(scene_name, scene_name + '.xml'), resx, resy, sensors)
    else:
        sensors_list = sensors[0](*sensors[1:])

    # Store a lambda function that allows to create sensors as we need them
    return (lambda: config_class(name, sensors=sensors_list,
                                 resx=resx, resy=resy, **kwargs), name)


def process_config_dicts(configs):
    """Takes a list of config dictionary, resolves parent-child dependencies
        and adds them to the config list"""
    assert len({c['name'] for c in configs}) == len(configs), "Each config name has to be unique!"
    name_map = {c['name']: c for c in configs}
    output_dicts = []
    for c in configs:
        current = c
        children = []
        while 'parent' in current:
            children.append(current)
            current = name_map[current['parent']]
            assert not current in children, "Circular dependency is not allowed!"

        final = dict(current)
        for child in reversed(children):
            for k in child:
                final[k] = child[k]
        if 'parent' in final:
            final.pop('parent')
        output_dicts.append(final)
    return output_dicts


CONFIG_DICTS = [
    {
        # Some common settings for all experiments
        'name': 'base',
        'config_class': SdfConfig,
        'sensors': (get_regular_cameras, 6),
        'sdf_regularizer_weight': 1e-5,
        'sdf_regularizer': reg.eval_discrete_laplacian_reg,
        'loss': losses.multiscale_l1,
        'upsample_iter': [64, 128],
        'sdf_res': 64,
        'resx': 128, 'resy': 128,
        'param_keys': [SDF_DEFAULT_KEY],
        'param_averaging_beta': 0.95,
    }, {
        'name': 'no-tex-6',
        'parent': 'base',
        'sensors': (get_regular_cameras, 6),
        'use_multiscale_rendering': True,
        'render_upsample_iter': [180],
        'upsample_iter': [64, 128, 180],
        'sdf_res': 64,
        'resx': 128, 'resy': 128,
        'param_keys': [SDF_DEFAULT_KEY],
    }, {
        'name': 'no-tex-12',
        'parent': 'no-tex-6',
        'use_multiscale_rendering': False,
        'sensors': (get_regular_cameras, 12),
        'upsample_iter': [64, 128],
        'batch_size': 6
    }, {
        'name': 'torus-shadow-1',
        'parent': 'no-tex-12',
        'scene_name': 'torus-shadow',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220],
        'upsample_iter': [128, 140, 180, 220],
        'sdf_res': 128,
        'resx': 256, 'resy': 256,
        'sensors': [0] # Use one sensor that is directly from the scene
    }, {
        'name': 'mirror-opt-1',
        'parent': 'no-tex-12',
        'scene_name': 'mirror-opt',
        'upsample_iter': [128, 220],
        'sdf_res': 64,
        'resx': 128, 'resy': 128,
        'sensors': [0]
    }, {
        'name': 'mirror-opt-hq',
        'parent': 'no-tex-12',
        'scene_name': 'mirror-opt',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220],
        'upsample_iter': [128, 180, 220],
        'sdf_res': 128,
        'resx': 256, 'resy': 256,
        'sensors': [0]
    }, {
        'name': 'no-tex-3',
        'parent': 'no-tex-6',
        'sensors': (get_regular_cameras, 3)
    }, {
        'name': 'diffuse-6',
        'parent': 'base',
        'sensors': (get_regular_cameras, 6),
        'use_multiscale_rendering': False,
        'upsample_iter': [128, 180],
        'sdf_res': 64,
        'resx': 128, 'resy': 128,
        'param_keys': [SDF_DEFAULT_KEY, 'main-bsdf.reflectance.volume.data']
    }, {
        'name': 'principled-6',
        'parent': 'diffuse-6',
        'use_multiscale_rendering': False,
        'param_keys': [SDF_DEFAULT_KEY, 'main-bsdf.base_color.volume.data', 'main-bsdf.roughness.volume.data']
    }, {
        'name': 'diffuse-12',
        'parent': 'diffuse-6',
        'sensors': (get_regular_cameras, 12),
        'batch_size': 6,
    }, {
        'name': 'principled-12',
        'parent': 'principled-6',
        'sensors': (get_regular_cameras, 12),
        'batch_size': 6,
        'upsample_iter': [128, 180]
    }, {
        'name': 'diffuse-12-hq',
        'parent': 'diffuse-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220],
        'upsample_iter': [128, 180, 220],
        'sdf_res': 128,
        'resx': 256, 'resy': 256,
    }, {
        'name': 'diffuse-12-hqq',
        'parent': 'diffuse-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220, 300],
        'upsample_iter': [128, 180, 220, 270],
        'sdf_res': 256,
        'resx': 512, 'resy': 512,
    }, {
        'name': 'diffuse-16-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 16),
    }, {
        'name': 'diffuse-20-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 20),
    }, {
        'name': 'diffuse-32-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 32),
    }, {
        'name': 'diffuse-32-hqq-2',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 32),
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220, 400],
        'upsample_iter': [128, 180, 220, 450],
        'sdf_res': 256,
        'resx': 512, 'resy': 512,
    }, {
        'name': 'diffuse-40-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 40),
    }, {
        'name': 'diffuse-64-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 40),
    }, {
        'name': 'diffuse-24-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras, 24),
    }, {
        'name': 'diffuse-16-top-hq',
        'parent': 'diffuse-12-hq',
        'sensors': (get_regular_cameras_top, 16),
    }, {
        'name': 'diffuse-16-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras, 16),
    }, {
        'name': 'diffuse-24-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras, 24),
    }, {
        'name': 'diffuse-40-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras, 40),
    }, {
        'name': 'diffuse-48-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras, 48),
    }, {
        'name': 'diffuse-64-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras, 64),
    }, {
        'name': 'diffuse-16-top-hqq',
        'parent': 'diffuse-12-hqq',
        'sensors': (get_regular_cameras_top, 16),
    }, {
        'name': 'diffuse-16-hqq-2',
        'parent': 'diffuse-12-hqq',
        'render_upsample_iter': [300],
        'sdf_init_fn': lambda res: create_sphere_sdf(res, radius=0.1),
        'tex_upsample_iter': [120, 150, 180, 200, 300, 400],
        'sdf_regularizer_weight': 1e-4,
        'sdf_regularizer': reg.eval_discrete_laplacian_reg,
        'upsample_iter': [150, 180],
    }, {
        'name': 'diffuse-32-hqq',
        'parent': 'diffuse-16-hqq',
        'sensors': (get_regular_cameras, 32)
    }, {
        'name': 'diffuse-32-top-hqq',
        'parent': 'diffuse-16-hqq',
        'sensors': (get_regular_cameras_top, 32)
    }, {
        'name': 'no-tex-12-hq',
        'parent': 'no-tex-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220],
        'upsample_iter': [128, 180, 220],
        'sdf_res': 128,
        'resx': 256, 'resy': 256,
    }, {
        'name': 'no-tex-1-hq',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 1),
    }, {
        'name': 'no-tex-2-hq',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 2),
    }, {
        'name': 'no-tex-3-hq',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 3),
    }, {
        'name': 'no-tex-6-hq',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 6),
    }, {
        'name': 'no-tex-32-hq',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 32),
    }, {
        'name': 'no-tex-1',
        'parent': 'no-tex-12',
        'sensors': (get_regular_cameras, 1),
    }, {
        'name': 'no-tex-2',
        'parent': 'no-tex-12',
        'sensors': (get_regular_cameras, 2),
    }, {
        'name': 'no-tex-32',
        'parent': 'no-tex-12',
        'sensors': (get_regular_cameras, 32),
    }, {
        'name': 'no-tex-32-hq-l1',
        'parent': 'no-tex-32-hq',
        'loss': losses.l1
    }, {
        'name': 'no-tex-32-hq-mape',
        'parent': 'no-tex-32-hq',
        'loss': losses.mape
    }, {
        'name': 'no-tex-32-hq-no-reg',
        'parent': 'no-tex-32-hq',
        'sdf_regularizer_weight': 0.0,
        'loss': losses.l1
    }, {
        'name': 'no-tex-6-hqq',
        'parent': 'no-tex-6',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220, 300],
        'upsample_iter': [128, 180, 220, 270],
        'sdf_res': 256,
        'resx': 512, 'resy': 512,
    }, {
        'name': 'no-tex-12-hqq',
        'parent': 'no-tex-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220, 300],
        'upsample_iter': [128, 180, 220, 270],
        'sdf_res': 256,
        'resx': 512, 'resy': 512,
    }, {
        'name': 'no-tex-32-hqq',
        'parent': 'no-tex-12-hqq',
        'sensors': (get_regular_cameras, 32),
    }, {
        'name': 'principled-12-hq',
        'parent': 'principled-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220],
        'upsample_iter': [128, 180, 220],
        'sdf_res': 128,
        'resx': 256, 'resy': 256,
    }, {
        'name': 'principled-12-hqq',
        'parent': 'principled-12',
        'use_multiscale_rendering': True,
        'render_upsample_iter': [220, 300],
        'upsample_iter': [128, 180, 220, 270],
        'sdf_res': 256,
        'resx': 512, 'resy': 512,
    }, {
        'name': 'principled-16-hq',
        'parent': 'principled-12-hq',
        'sensors': (get_regular_cameras, 16),
    }, {
        'name': 'principled-16-hqq',
        'parent': 'principled-12-hqq',
        'sensors': (get_regular_cameras, 16),
    }, {
        'name': 'principled-32-hq',
        'parent': 'principled-16-hq',
        'sensors': (get_regular_cameras, 32)
    }, {
        'name': 'principled-32-hqq',
        'parent': 'principled-16-hqq',
        'sensors': (get_regular_cameras, 32)
    }, {
        'name': 'principled-48-hqq',
        'parent': 'principled-16-hqq',
        'sensors': (get_regular_cameras, 48)
    }, {
        'name': 'principled-64-hqq',
        'parent': 'principled-16-hqq',
        'sensors': (get_regular_cameras, 64)
    }
]

# Add shifted versions of no-tex config for "variance" figure
N_SHIFTS = 8
for shift in range(N_SHIFTS):
    CONFIG_DICTS.append({
        'name': f'no-tex-3-hq-{shift}',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 3, shift / N_SHIFTS),
    })
    CONFIG_DICTS.append({
        'name': f'no-tex-2-hq-{shift}',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 2, shift / N_SHIFTS),
    })
    CONFIG_DICTS.append({
        'name': f'no-tex-6-hq-{shift}',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 6, shift / N_SHIFTS),
    })
    CONFIG_DICTS.append({
        'name': f'no-tex-12-hq-{shift}',
        'parent': 'no-tex-12-hq',
        'sensors': (get_regular_cameras, 12, shift / N_SHIFTS),
    })

PROCESSED_SCENE_CONFIG_DICTS = process_config_dicts(CONFIG_DICTS)
for processed in PROCESSED_SCENE_CONFIG_DICTS:
    fn, name = create_scene_config_init_fn(**processed)
    SCENE_CONFIGS[name] = fn
del fn, name


def is_valid_opt_config(scene):
    return scene in SCENE_CONFIGS


def get_opt_config(scene, cmd_args=None):
    """Retrieve configuration options associated with a given scene"""
    if scene in SCENE_CONFIGS:
        if cmd_args is None:
            return SCENE_CONFIGS[scene]()
        else:
            # Somewhat involved logic to allow for command line arguments to override parameters
            # of the original config dict *and* the processed config object, plus returns any remaining args

            # 1. obtain the dict with the right config name
            d = [d for d in PROCESSED_SCENE_CONFIG_DICTS if d['name'] == scene][0]

            # 2. apply args to the dict
            cmd_args = apply_cmdline_args(d, cmd_args)

            # 3. Instantiate the actual config
            config = create_scene_config_init_fn(**d)[0]()

            # 4. Potentially apply args to the config after instantiation too (mightr be redundant)
            cmd_args = apply_cmdline_args(config, cmd_args)
            return config, cmd_args
    else:
        raise ValueError("Invalid scene config name!")
