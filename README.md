# Electromagnetic System

Implementation of [finite-difference time-domain method](https://ieeexplore.ieee.org/document/1138693/) to model electromagnetic field propagation in Python, with CUDA support. This is my final project for the computational lab [PHYC20013](https://handbook.unimelb.edu.au/2020/subjects/phyc20013) at the University of Melbourne in 2020.

## Files

* `emsystem.py`: Classes and functions for CPU use.
* `emsystem_gpu.py`: Classes and functions for CUDA use.
  * Note: Only support GPUs with CUDA capability.
* `Examples.ipynb`: Example applications

## Preparation

The project depends on the following Python packages:

* numpy
* matplotlib
* (for CUDA) numba

The following software is required for quality animation generation:

* ImageMagick
  * Download: <https://imagemagick.org/script/download.php>
  * **Remember to check the option 'Install legacy utilities (e.g. convert)' during the stage 'Select Additional Tasks' when installing**

## Examples

Below are some animations made in `Examples.ipynb`. The electric and magnetic fields are represented with red and blue quivers respectively.

### Propagation of a plane electromagnetic pulse

![Propagation of a plane electromagnetic pulse](gallery/plane-pulse-propagation-gpu.gif)

### Reflection of electromagentic pulse on other material

![Reflection of electromagentic pulse on other material](gallery/plane-pulse-reflection-gpu.gif)

### Propagation of a point electromagnetic pulse

![Propagation of a point electromagnetic pulse](gallery/point-pulse-propagation-gpu.gif)
