"""Density-adaptive non-rigid registration for initially aligned ALS--MLS data.

This reference implementation uses a regular 3-D control grid, piecewise
trilinear displacement interpolation, Cauchy and normal-consistency
correspondence weights, support-adaptive Tikhonov regularization, and
coordinate-wise conjugate-gradient solves.

ALS is treated as the fixed target and MLS as the moving source.  The method is
intended for fine registration after an initial alignment; it does not perform
coarse registration.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg


LOGGER = logging.getLogger("density_adaptive_registration")


@dataclass(frozen=True)
class RegistrationConfig:
    """Numerical settings used in the experiments reported in the manuscript."""

    voxel_size: float = 5.0
    max_correspondence_distance: float = 4.0
    alpha: float = 1.0
    beta: float = 4.0
    c0: float = 1.0
    gamma: float = 0.6
    c_min: float = 0.1
    normal_dot_threshold: float = 0.707
    damping: float = 1.0e-5
    cg_rtol: float = 1.0e-5
    cg_max_iterations: int = 300
    outer_iterations: int = 6
    normal_radius: float = 2.0
    normal_max_nn: int = 30
    normal_orientation_point: tuple[float, float, float] = (0.0, 0.0, 1000.0)

    def validate(self) -> None:
        if self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if self.max_correspondence_distance <= 0.0:
            raise ValueError("max_correspondence_distance must be positive")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("alpha and beta must be nonnegative")
        if self.c0 <= 0.0 or self.c_min <= 0.0:
            raise ValueError("c0 and c_min must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not -1.0 <= self.normal_dot_threshold <= 1.0:
            raise ValueError("normal_dot_threshold must be in [-1, 1]")
        if self.damping < 0.0 or self.cg_rtol <= 0.0:
            raise ValueError("damping must be nonnegative and cg_rtol must be positive")
        if self.cg_max_iterations <= 0 or self.outer_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        if self.normal_radius <= 0.0 or self.normal_max_nn <= 0:
            raise ValueError("normal-estimation settings must be positive")


class DensityAdaptiveTrilinearRegistration:
    """Piecewise-trilinear non-rigid fine registration.

    Candidate-correspondence participation determines the node-support measure
    used by the adaptive regularizer.  This measure is not the physical point
    density (points per square metre) of either input cloud.
    """

    _OFFSETS = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int64,
    )

    def __init__(self, config: RegistrationConfig | None = None) -> None:
        self.config = config or RegistrationConfig()
        self.config.validate()

    @staticmethod
    def _copy_point_cloud(
        point_cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        return o3d.geometry.PointCloud(point_cloud)

    def _estimate_normals(self, point_cloud: o3d.geometry.PointCloud) -> None:
        cfg = self.config
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=cfg.normal_radius,
                max_nn=cfg.normal_max_nn,
            )
        )
        point_cloud.orient_normals_towards_camera_location(
            np.asarray(cfg.normal_orientation_point, dtype=np.float64)
        )

    @staticmethod
    def _robust_weights(
        source_points: np.ndarray,
        target_points: np.ndarray,
        source_normals: np.ndarray,
        target_normals: np.ndarray,
        c_scale: float,
        normal_dot_threshold: float,
    ) -> np.ndarray:
        """Return Cauchy residual weights multiplied by normal weights."""
        coordinate_difference = source_points - target_points
        point_to_plane_residual = np.sum(
            coordinate_difference * target_normals,
            axis=1,
        )
        cauchy_weights = 1.0 / (
            1.0 + np.square(point_to_plane_residual / c_scale)
        )

        normal_consistency = np.sum(source_normals * target_normals, axis=1)
        accepted = normal_consistency > normal_dot_threshold
        normal_weights = np.square(np.clip(normal_consistency, 0.0, 1.0))
        return cauchy_weights * normal_weights * accepted.astype(np.float64)

    @staticmethod
    def _build_grid_topology(
        grid_shape: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Construct the six-neighbour grid-difference operator.

        Node indices use NumPy C-order throughout, matching
        ``numpy.ravel_multi_index`` in the interpolation matrix.
        """
        shape = tuple(int(value) for value in grid_shape)
        coordinates = np.indices(shape, dtype=np.int64).reshape(3, -1).T
        first_endpoints: list[np.ndarray] = []
        second_endpoints: list[np.ndarray] = []

        for axis in range(3):
            base = coordinates[coordinates[:, axis] < shape[axis] - 1]
            neighbour = base.copy()
            neighbour[:, axis] += 1
            first_endpoints.append(np.ravel_multi_index(base.T, shape))
            second_endpoints.append(np.ravel_multi_index(neighbour.T, shape))

        edge_u = np.concatenate(first_endpoints)
        edge_v = np.concatenate(second_endpoints)
        edge_count = edge_u.size
        rows = np.repeat(np.arange(edge_count, dtype=np.int64), 2)
        cols = np.column_stack((edge_u, edge_v)).reshape(-1)
        data = np.tile(np.asarray([1.0, -1.0]), edge_count)
        difference_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(edge_count, int(np.prod(grid_shape))),
        )
        return difference_matrix, edge_u, edge_v

    @staticmethod
    def _trilinear_components(
        points: np.ndarray,
        minimum_bound: np.ndarray,
        voxel_size: float,
        grid_shape: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized = (points - minimum_bound) / voxel_size
        cell_indices = np.floor(normalized).astype(np.int64)
        cell_indices = np.clip(cell_indices, 0, grid_shape - 2)
        local_coordinates = np.clip(normalized - cell_indices, 0.0, 1.0)
        return cell_indices, local_coordinates

    @classmethod
    def _interpolation_matrix(
        cls,
        cell_indices: np.ndarray,
        local_coordinates: np.ndarray,
        grid_shape: np.ndarray,
    ) -> tuple[sp.csr_matrix, np.ndarray]:
        match_count = cell_indices.shape[0]
        node_count = int(np.prod(grid_shape))
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        values: list[np.ndarray] = []
        node_support_counts = np.zeros(node_count, dtype=np.float64)
        shape = tuple(int(value) for value in grid_shape)

        for offset in cls._OFFSETS:
            node_coordinates = cell_indices + offset
            node_indices = np.ravel_multi_index(node_coordinates.T, shape)
            axis_weights = np.where(
                offset[np.newaxis, :] == 0,
                1.0 - local_coordinates,
                local_coordinates,
            )
            interpolation_weights = np.prod(axis_weights, axis=1)

            rows.append(np.arange(match_count, dtype=np.int64))
            cols.append(node_indices)
            values.append(interpolation_weights)

            # The manuscript defines support by candidate participation.  The
            # count therefore includes candidates whose final robust weight is 0.
            np.add.at(node_support_counts, node_indices, 1.0)

        matrix = sp.csr_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(cols)),
            ),
            shape=(match_count, node_count),
        )
        return matrix, node_support_counts

    @classmethod
    def _interpolate_node_displacements(
        cls,
        node_displacements: np.ndarray,
        cell_indices: np.ndarray,
        local_coordinates: np.ndarray,
        grid_shape: np.ndarray,
    ) -> np.ndarray:
        point_displacements = np.zeros(
            (cell_indices.shape[0], 3),
            dtype=np.float64,
        )
        shape = tuple(int(value) for value in grid_shape)
        for offset in cls._OFFSETS:
            node_coordinates = cell_indices + offset
            node_indices = np.ravel_multi_index(node_coordinates.T, shape)
            axis_weights = np.where(
                offset[np.newaxis, :] == 0,
                1.0 - local_coordinates,
                local_coordinates,
            )
            interpolation_weights = np.prod(axis_weights, axis=1)
            point_displacements += (
                node_displacements[node_indices]
                * interpolation_weights[:, np.newaxis]
            )
        return point_displacements

    def register(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        """Register moving MLS ``source`` to fixed ALS ``target``."""
        cfg = self.config
        if source.is_empty() or target.is_empty():
            raise ValueError("source and target point clouds must be nonempty")

        current_source = self._copy_point_cloud(source)
        fixed_target = self._copy_point_cloud(target)
        self._estimate_normals(current_source)
        self._estimate_normals(fixed_target)

        target_bbox = fixed_target.get_axis_aligned_bounding_box()
        minimum_bound = target_bbox.get_min_bound() - 2.0 * cfg.voxel_size
        maximum_bound = target_bbox.get_max_bound() + 2.0 * cfg.voxel_size
        grid_shape = (
            np.ceil((maximum_bound - minimum_bound) / cfg.voxel_size)
            .astype(np.int64)
            + 1
        )
        node_count = int(np.prod(grid_shape))
        difference_matrix, edge_u, edge_v = self._build_grid_topology(grid_shape)

        LOGGER.info(
            "Control grid: shape=%s, nodes=%d, adjacency edges=%d",
            grid_shape.tolist(),
            node_count,
            difference_matrix.shape[0],
        )

        target_points = np.asarray(fixed_target.points)
        target_normals = np.asarray(fixed_target.normals)
        target_tree = o3d.geometry.KDTreeFlann(fixed_target)

        for outer_iteration in range(cfg.outer_iterations):
            iteration_start = time.perf_counter()
            c_scale = max(cfg.c_min, cfg.c0 * cfg.gamma**outer_iteration)
            source_points = np.asarray(current_source.points)
            source_normals = np.asarray(current_source.normals)

            source_indices: list[int] = []
            target_indices: list[int] = []
            maximum_squared_distance = cfg.max_correspondence_distance**2
            for source_index, point in enumerate(source_points):
                neighbour_count, indices, squared_distances = (
                    target_tree.search_knn_vector_3d(point, 1)
                )
                if (
                    neighbour_count > 0
                    and squared_distances[0] < maximum_squared_distance
                ):
                    source_indices.append(source_index)
                    target_indices.append(indices[0])

            if not source_indices:
                raise RuntimeError(
                    "No candidate correspondences satisfy the distance threshold"
                )
            if len(source_indices) < 100:
                LOGGER.warning(
                    "Only %d candidate correspondences were retained",
                    len(source_indices),
                )

            matched_source = source_points[source_indices]
            matched_target = target_points[target_indices]
            matched_source_normals = source_normals[source_indices]
            matched_target_normals = target_normals[target_indices]

            robust_weights = self._robust_weights(
                matched_source,
                matched_target,
                matched_source_normals,
                matched_target_normals,
                c_scale,
                cfg.normal_dot_threshold,
            )
            weight_matrix = sp.diags(robust_weights, format="csr")

            cell_indices, local_coordinates = self._trilinear_components(
                matched_source,
                minimum_bound,
                cfg.voxel_size,
                grid_shape,
            )
            interpolation_matrix, node_support_counts = (
                self._interpolation_matrix(
                    cell_indices,
                    local_coordinates,
                    grid_shape,
                )
            )

            maximum_support = float(np.max(node_support_counts))
            normalized_support = node_support_counts / (maximum_support + 1.0e-6)
            normalized_support = np.clip(normalized_support, 0.0, 1.0)
            node_regularization = cfg.alpha * np.exp(
                cfg.beta * (1.0 - normalized_support)
            )
            edge_regularization = np.maximum(
                node_regularization[edge_u],
                node_regularization[edge_v],
            )
            edge_weight_matrix = sp.diags(edge_regularization, format="csr")

            system_matrix = (
                interpolation_matrix.T
                @ weight_matrix
                @ interpolation_matrix
                + difference_matrix.T
                @ edge_weight_matrix
                @ difference_matrix
                + cfg.damping * sp.eye(node_count, format="csr")
            ).tocsr()
            coordinate_differences = matched_target - matched_source
            node_updates = np.zeros((node_count, 3), dtype=np.float64)

            for coordinate in range(3):
                right_hand_side = (
                    interpolation_matrix.T
                    @ weight_matrix
                    @ coordinate_differences[:, coordinate]
                )
                solution, solver_info = splinalg.cg(
                    system_matrix,
                    right_hand_side,
                    rtol=cfg.cg_rtol,
                    maxiter=cfg.cg_max_iterations,
                )
                if solver_info < 0:
                    raise RuntimeError(
                        f"CG failed for coordinate {coordinate} (info={solver_info})"
                    )
                if solver_info > 0:
                    LOGGER.warning(
                        "CG reached its iteration limit for coordinate %d "
                        "(info=%d)",
                        coordinate,
                        solver_info,
                    )
                node_updates[:, coordinate] = solution

            all_cell_indices, all_local_coordinates = self._trilinear_components(
                source_points,
                minimum_bound,
                cfg.voxel_size,
                grid_shape,
            )
            point_updates = self._interpolate_node_displacements(
                node_updates,
                all_cell_indices,
                all_local_coordinates,
                grid_shape,
            )
            current_source.points = o3d.utility.Vector3dVector(
                source_points + point_updates
            )
            self._estimate_normals(current_source)

            elapsed = time.perf_counter() - iteration_start
            maximum_update = float(np.max(np.linalg.norm(point_updates, axis=1)))
            LOGGER.info(
                "Iteration %d/%d: candidates=%d, nonzero weights=%d, "
                "c=%.4f, max update=%.6f m, time=%.2f s",
                outer_iteration + 1,
                cfg.outer_iterations,
                len(source_indices),
                int(np.count_nonzero(robust_weights)),
                c_scale,
                maximum_update,
                elapsed,
            )

        return current_source


def _three_floats(values: Sequence[str]) -> tuple[float, float, float]:
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3:
        raise ValueError("exactly three values are required")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-register an initially aligned MLS point cloud (moving source) "
            "to an ALS point cloud (fixed target)."
        )
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="moving MLS file"
    )
    parser.add_argument(
        "--target", required=True, type=Path, help="fixed ALS file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="registered output file"
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=5.0,
        help="control-grid spacing in metres",
    )
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=4.0,
        help="candidate distance threshold in metres",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="base Tikhonov coefficient"
    )
    parser.add_argument(
        "--beta", type=float, default=4.0, help="support-adaptation coefficient"
    )
    parser.add_argument("--c0", type=float, default=1.0, help="initial Cauchy scale")
    parser.add_argument(
        "--gamma", type=float, default=0.6, help="Cauchy-scale decay factor"
    )
    parser.add_argument("--c-min", type=float, default=0.1, help="minimum Cauchy scale")
    parser.add_argument(
        "--normal-dot-threshold",
        type=float,
        default=0.707,
        help="hard normal-consistency threshold",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=1.0e-5,
        help="fixed identity damping",
    )
    parser.add_argument(
        "--cg-rtol",
        type=float,
        default=1.0e-5,
        help="CG relative tolerance",
    )
    parser.add_argument(
        "--cg-max-iterations",
        type=int,
        default=300,
        help="maximum CG iterations",
    )
    parser.add_argument(
        "--outer-iterations",
        type=int,
        default=6,
        help="outer reweighting iterations",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=2.0,
        help="normal-estimation radius in metres",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=30,
        help="maximum normal-estimation neighbours",
    )
    parser.add_argument(
        "--normal-orientation-point",
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 1000.0),
        help="viewpoint used to orient normals (default: 0 0 1000)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="enable detailed logging"
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.source.is_file():
        raise FileNotFoundError(f"source file not found: {args.source}")
    if not args.target.is_file():
        raise FileNotFoundError(f"target file not found: {args.target}")

    source_cloud = o3d.io.read_point_cloud(str(args.source))
    target_cloud = o3d.io.read_point_cloud(str(args.target))
    if source_cloud.is_empty():
        raise ValueError(f"failed to read a nonempty source cloud: {args.source}")
    if target_cloud.is_empty():
        raise ValueError(f"failed to read a nonempty target cloud: {args.target}")

    config = RegistrationConfig(
        voxel_size=args.voxel_size,
        max_correspondence_distance=args.max_correspondence_distance,
        alpha=args.alpha,
        beta=args.beta,
        c0=args.c0,
        gamma=args.gamma,
        c_min=args.c_min,
        normal_dot_threshold=args.normal_dot_threshold,
        damping=args.damping,
        cg_rtol=args.cg_rtol,
        cg_max_iterations=args.cg_max_iterations,
        outer_iterations=args.outer_iterations,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        normal_orientation_point=_three_floats(args.normal_orientation_point),
    )
    registered_cloud = DensityAdaptiveTrilinearRegistration(config).register(
        source_cloud,
        target_cloud,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(
        str(args.output),
        registered_cloud,
        write_ascii=False,
    ):
        raise OSError(f"failed to write output point cloud: {args.output}")
    LOGGER.info("Registered point cloud written to %s", args.output)


if __name__ == "__main__":
    main()
