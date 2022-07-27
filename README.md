<!-- PROJECT LOGO -->
<p align="center">

  <h1 align="center"><a href="https://dvicini.github.io/differentiable-sdf-rendering/">Differentiable Signed Distance Function Rendering</a></h1>

![Teaser](./teaser_dark.png#gh-dark-mode-only)
![Teaser](./teaser.png#gh-light-mode-only)

  <p align="center">
    ACM Transactions on Graphics (Proceedings of SIGGRAPH), July 2022.
    <br />
    <a href="https://dvicini.github.io/"><strong>Delio Vicini</strong></a>
    ·
    <a href="https://speierers.github.io/"><strong>Sébastien Speierer</strong></a>
    ·
    <a href="https://rgl.epfl.ch/people/wjakob"><strong>Wenzel Jakob</strong></a>
  </p>

  <p align="center">
    <a href='http://rgl.s3.eu-central-1.amazonaws.com/media/papers/Vicini2022sdf_1.pdf'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat-square' alt='Paper PDF'>
    </a>
    <a href='https://dvicini.github.io/differentiable-sdf-rendering/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat-square' alt='Project Page'>
    </a>
  </p>
</p>

## About
This repository contains the Python code to reproduce some of the experiments of the Siggraph 2022 paper "Differentiable Signed Distance Function Rendering". Please see the [project page](https://dvicini.github.io/differentiable-sdf-rendering/) for the paper and supplemental video.

The code in this repository can be used to compare to this paper or as a starting point for improved/alternative versions of differentiable SDF rendering. The project structure might also serve as an example of how to build a highly customized optimization pipeline using Mitsuba 3.

This repository also contains the majority of figure generation scrips we used to generate results for our paper. Those could be useful if you want to generate similar figures.

**Note**: All experiments in this paper were done on *synthetic* data, which means that this repository is not useful as is to reconstruct objects from captured images. This code is meant for researchers working on related problems, *not* end users.

## Installation

The code in this repository depends on the [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) and [Dr.Jit](https://github.com/mitsuba-renderer/drjit) Python modules. Additionally,
we provide a separate Python module that re-distances an SDF using the fast sweeping method.
This package is called [fastsweep](https://github.com/rgl-epfl/fastsweep) and can also be installed using `pip`.

The code further depends on a few common Python packages (`numpy`, `matplotlib`, etc.).
The following command installs all the required dependencies:

```bash
pip install mitsuba fastsweep numpy tqdm matplotlib
```
Both the `mitsuba` and `fastsweep` module might receive updates and bug fixes. You can install the latest versions of these modules using `pip install -U mitsuba fastsweep`.


Some of the figures further require [scikit-fmm](https://pythonhosted.org/scikit-fmm/) to re-distance 2D SDFs and some common data visualization packages:
```bash
pip install seaborn pandas scikit-fmm
```
The code was mainly used on Linux (Ubuntu) but was also (partially) tested on Windows and macOS. It can run on the CPU using Mitsuba's LLVM backend, but the performance will be less good than when using a CUDA-capable GPU. Most results for the paper were generated using an Nvidia RTX Titan GPU (24 GB of memory). If you run out of GPU memory, you might need to reduce the resolution of the optimized SDFs.

### Cloning the repository and downloading the scenes

The rest of the setup is then to simply clone the repository
```bash
git clone https://github.com/rgl-epfl/differentiable-sdf-rendering.git
```

and download the example scenes from
[here](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Vicini2022SDF_1.zip).
This archive has to be extracted at the top level of this repository.

## Running an optimization
The primary entry point for the code in this repository is the file `optimize.py` in the `python` folder. It takes a scene and optimization configuration and runs a differentiable rendering optimization. This is the main script that you will want to use when experimenting with this code. By default, it will write all results in a directory `differentiable-sdf-rendering/output`.

An example invocation of this script would be
```
python optimize.py dragon --optconfig no-tex-12
```
This optimizes the `dragon` scene using the `no-tex-12` configuration (`no-tex` = we don't optimize any surface textures and `12` is the number of views). Optimization configurations with `-hq` and `-hqq` suffixes use higher SDF resolutions. See the file `opt_configs.py` for a list of available configurations.
The scene name is not a path, but rather the name of the scene folder in the `scenes` directory.

If everything is set up correctly, the above command will optimize an SDF approximating the Stanford dragon and write all of its results into `outputs/dragon/no-tex-12/warp/`. If `ffmpeg` is installed on the system, it will also generate basic convergence and turntable videos.


The script also supports overriding parameters such as the iteration count or name of the output folder:
```
python optimize.py dragon --optconfig no-tex-12 --n_iter=32 --name=myfolder
```
This is useful when trying to quickly experiment with different settings. Any parameter that's of a basic type (e.g., `int`, `string`, etc.) can be overwritten in this way.

## Generating figures
The repository also contains the pipeline to regenerate most of the figures from the paper. All the figure-related code is in the `figures` folder and each figure has a dedicated subfolder. For most figures, there is a python script `figures/figure_name/figure_name.py` that
generates the results used in the figure. This might involve running several optimizations and can therefore take a while. The jupyter notebook `figures/figure_name/figure_name.ipynb` then reads the generated results (if there are any) and generates the actual figure using `matplotlib`. The Jupyter notebooks do not run any expensive computations themselves and should all run in a few seconds. At the bottom of each notebook, there is a commented out command to export the figure as a PDF.

Note that due to the stochastic nature of the optimizations and refactoring of the code base the results might slightly differ from the ones in the paper. Generally, the results will be of similar quality to the ones shown in the paper, but do not match exactly.

The most interesting script is `figures/opt_results/opt_results.py`, as it regenerates the main result figure in the paper.

The following figures from the paper can be generated using the provided scripts:
- Figure 1: `figures/teaser`
- Figure 2: `figures/sphere_reparam` (for the paper we added a few extra annotations using Adobe Illustrator)
- Figure 3: `figures/sdf`
- Figure 4: `figures/normalization`
- Figure 5: `figures/diff_sphere_tracing`
- Figure 6: `figures/sphere_tracing_weights_ablation`
- Figure 7: `figures/nested_reparam`
- Figure 8: `figures/forwardgrads`
- Figure 9: `figures/benchmark`
- Figure 10: `figures/opt_results`
- Figure 11: `figures/conv_opt_results`
- Figure 13: `figures/indirect_reparam`
- Figure 15: `figures/shading_gradients`
- Figure 16: `figures/limitations`


There are two figures for which we do not provide the code currently:
- Figure 12: This figure shows a plot of the reconstruction variance and requires running 64 reconstructions. This takes quite a long time and is impractical to re-run (for the paper, we generated the results for this figure on multiple machines in parallel)
- Figure 14: The comparison of the variance requires disabling gradient propagation through the pixel filter normalization, which is currently not supported using the pre-built Mitsuba 3 Python module.

There is also no support for re-generating the supplemental video, as this process was not fully automated.

The `run_all_figures.py` script generates all the results used by the Jupyter notebooks to produce the final figures. This script is here for completeness, but takes a long time to run (>24 h) and outputs a lot of data (>30 GB).

## Code structure
The code in this repository is structured as follows:
- `python/` contains all the main scripts to run optimizations and compute gradients.
- `python/integrators/` contains various differentiable SDF integrators. For most results, we use a differentiable direct illumination integrator (`sdf_direct_reparam`). The optimization of the bunny visible in a reflection uses the `sdf_prb_reparam` integrator.
- `figures/` contains all the figure generation scripts. `figures/common.py` contains useful common functions and settings (e.g., font type) used across figures.

More specifically, some interesting files are:
- `constants.py` defines various default paths used throughout the code base.
- `config.py` defines configurations for different gradient computation methods
- `opt_configs.py` defines different optimization configurations (e.g., defining virtual sensors, parameter resolution, learning rates, etc.)
- `shape_opt.py` contains the main optimization loop
- `variables.py` abstracts the logic to initialize, update and read/write optimized parameters
- `shapes.py` contains the `Grid3d` class which stores our SDFs and implements the ray intersection routine
- `warp.py` defines our reparameterization and also implements the baseline we compare to in the paper.
- `mesh_to_sdf` provides a routine to convert a watertight mesh to an SDF.

## Limitations / future work
The SDF sphere tracing is currently implemented entirely in Python. This
means that SDFs are not an actual Mitsuba `Shape` plugin, which makes
some of the scene handling somewhat awkward (you might notice some "dummy" shape placeholders in scene files). Shapes are the only plugin class in Mitsuba 3 that currently cannot be implemented in Python, but this will potentially change in the future. This has the potential to reduce (OptiX) compilation times and also might improve performance.

For most of our experiments, we use scenes that only contain an SDF and no other objects. This means we can compile raw CUDA PTX instead of OptiX, which leads to lower compilation times and better overall performance.

Lastly, `prb_sdf_reparam.py` is largely copied from Mitsuba 3. Potentially some of this redundancy can be eliminated by making some of the reparameterization code in Mitsuba a bit more general.

## License
This code is provided under a 3-clause BSD license that can be found in the
LICENSE file. By using, distributing, or contributing to this project, you agree
to the terms and conditions of this license.

## Citation
If you use parts of this code for your own research, please consider citing our Siggraph paper:
```bibtex
@article{Vicini2022sdf,
    title   = {Differentiable Signed Distance Function Rendering},
    author  = {Delio Vicini and Sébastien Speierer and Wenzel Jakob},
    year    = 2022,
    month   = jul,
    journal = {Transactions on Graphics (Proceedings of SIGGRAPH)},
    volume  = 41,
    number  = 4,
    pages   = {125:1--125:18},
    doi     = {10.1145/3528223.3530139}
}
```
