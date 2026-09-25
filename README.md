# Density-Adaptive Non-Rigid Registration Using a Piecewise Trilinear Deformation Field

This repository provides a Python implementation of the density-adaptive non-rigid registration method described in the manuscript:

> **Density-Adaptive Nonrigid Registration of Airborne and Mobile Laser Scanning Point Clouds Using Piecewise Trilinear Deformation Fields**

The method is designed for fine registration of initially aligned airborne laser scanning (ALS) and mobile laser scanning (MLS) point clouds. ALS is treated as the fixed target point cloud, and MLS is treated as the moving source point cloud.

## Method Overview

The implementation consists of the following components:

* a regular three-dimensional voxel control grid;
* piecewise trilinear displacement interpolation from the eight control nodes of each voxel;
* Cauchy-type soft down-weighting based on point-to-plane residuals;
* hard screening and continuous weighting based on normal consistency;
* support-adaptive Tikhonov regularization based on candidate-correspondence participation at each control node;
* a fixed small identity-damping term for numerical stabilization; and
* separate conjugate-gradient solutions for the x, y, and z displacement components.

The node-support measure used by the adaptive regularizer is derived from the number of candidate correspondences involving each control node. It is not the physical sampling density of either point cloud and is not the ALS-to-MLS density ratio.

The point-to-plane residual is used to evaluate and weight candidate correspondences. The final data term fits weighted three-dimensional coordinate differences; therefore, the method is not a strictly point-to-plane least-squares model.

## Scope

This implementation assumes that the source and target point clouds are already approximately aligned. It does not perform coarse registration, absolute georeferencing, sensor calibration, or scene-change detection.

The trilinear deformation field and adjacent-node displacement regularization support local propagation and smooth variation of displacement values. They do not explicitly enforce first-derivative continuity, local rigidity, façade planarity, deformation invertibility, or complete topology preservation.

The repository currently provides the core proposed registration method. Dataset preparation, semi-synthetic perturbation generation, baseline implementations, and M3C2 evaluation are not included in the core registration script.

## Repository Contents

* `density_adaptive_registration.py`: implementation of the proposed registration method;
* `requirements.txt`: Python dependencies;
* `.gitignore`: files excluded from version control; and
* `README.md`: installation and usage instructions.

## Requirements

Python 3.10 or later is recommended.

The implementation requires:

* NumPy;
* SciPy; and
* Open3D.

Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

Alternatively, install them directly:

```bash
python -m pip install numpy scipy open3d
```

For reproducibility, the versions used to generate the reported experimental results should be recorded in `requirements.txt`.

## Usage

The basic command is:

```bash
python density_adaptive_registration.py \
  --source path/to/mls_source.ply \
  --target path/to/als_target.ply \
  --output path/to/registered_mls.ply \
  --verbose
```

The arguments have the following meanings:

* `--source`: initially aligned MLS point cloud to be deformed;
* `--target`: fixed ALS target point cloud;
* `--output`: output path for the registered MLS point cloud; and
* `--verbose`: prints iteration-level information.

On Windows, the program can be executed in one line:

```bash
python density_adaptive_registration.py --source "D:\data\mls_source.ply" --target "D:\data\als_target.ply" --output "D:\results\registered_mls.ply" --verbose
```

The input and output point-cloud formats must be supported by Open3D.

## Default Parameters

The default settings correspond to the principal parameter configuration described in the manuscript.

| Parameter                       | Command-line option             | Default value | Description                                      |
| ------------------------------- | ------------------------------- | ------------: | ------------------------------------------------ |
| Control-grid spacing            | `--voxel-size`                  |         5.0 m | Spacing between adjacent control nodes           |
| Maximum correspondence distance | `--max-correspondence-distance` |         4.0 m | Distance threshold for candidate correspondences |
| Base regularization coefficient | `--alpha`                       |           1.0 | Base Tikhonov coefficient                        |
| Support-adaptation coefficient  | `--beta`                        |           4.0 | Controls regularization adaptation               |
| Initial Cauchy scale            | `--c0`                          |           1.0 | Initial robust scale                             |
| Scale-decay factor              | `--gamma`                       |           0.6 | Decay factor between outer iterations            |
| Minimum Cauchy scale            | `--c-min`                       |           0.1 | Lower bound of the robust scale                  |
| Normal-consistency threshold    | `--normal-dot-threshold`        |         0.707 | Hard screening threshold                         |
| Fixed identity damping          | `--damping`                     |        1×10⁻⁵ | Numerical stabilization term                     |
| CG relative tolerance           | `--cg-rtol`                     |        1×10⁻⁵ | Relative convergence tolerance                   |
| Maximum CG iterations           | `--cg-max-iterations`           |           300 | Maximum iterations per coordinate solve          |
| Outer iterations                | `--outer-iterations`            |             6 | Correspondence and weight update iterations      |
| Normal-estimation radius        | `--normal-radius`               |         2.0 m | Neighborhood radius for normal estimation        |
| Maximum normal neighbors        | `--normal-max-nn`               |            30 | Maximum neighborhood size                        |

All parameters can be changed through the corresponding command-line options. When reproducing the experiments reported in the manuscript, the default principal configuration should be used unless otherwise stated.

## Example

```bash
python density_adaptive_registration.py \
  --source data/mls_scene.ply \
  --target data/als_scene.ply \
  --output outputs/mls_scene_registered.ply \
  --voxel-size 5.0 \
  --max-correspondence-distance 4.0 \
  --outer-iterations 6 \
  --verbose
```

The output is the deformed MLS point cloud in the coordinate system of the fixed ALS point cloud.

## Data Availability

The experiments described in the manuscript use six ALS–MLS scenes from the publicly available WHU-Urban3D dataset:

https://whu3d.com/

Users should obtain the dataset from its official source and comply with its terms of use. The original point-cloud data are not redistributed through this repository.

The semi-synthetic perturbations described in the manuscript are applied only to the MLS point clouds, while the ALS point clouds remain fixed.

## Reproducibility Note

The registration code uses candidate-correspondence participation to calculate control-node support. Candidate correspondences that subsequently receive a zero robust weight are still included in this support count, consistent with the formulation described in the manuscript.

The grid-interpolation and grid-adjacency operators use the same NumPy C-order node indexing. The source normals, correspondences, robust weights, node-support measures, and regularization coefficients are updated during the outer iterations.

Runtime depends on the number of source points, scene extent, control-grid resolution, candidate-correspondence distribution, hardware configuration, and software environment. The reported runtime in the manuscript should not be interpreted as end-to-end or real-time performance.

## Citation

If you use this implementation, please cite the associated manuscript:

```text
Density-Adaptive Nonrigid Registration of Airborne and Mobile Laser
Scanning Point Clouds Using Piecewise Trilinear Deformation Fields.
Manuscript submitted to the Journal of Applied Remote Sensing.
```

The complete bibliographic information and DOI will be added after publication.

## Contact

Questions and implementation issues may be submitted through the GitHub Issues page of this repository.

