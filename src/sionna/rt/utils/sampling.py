#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for sampling random points and computing signal strength"""

from __future__ import annotations
from typing import Tuple, Dict, List, TYPE_CHECKING
import mitsuba as mi
import drjit as dr
import numpy as np

if TYPE_CHECKING:
    from sionna.rt import Scene


def sample_random_positions(
    scene: 'Scene',
    num_samples: int,
    z_height: float | None = None,
    seed: int = 42,
    bounds: Tuple[float, float, float, float] | None = None
) -> mi.Point3f:
    """
    Sample random positions within the scene bounds.
    
    :param scene: Scene for which to sample positions
    :param num_samples: Number of random positions to sample
    :param z_height: Fixed z-coordinate for all sampled positions. 
        If None, samples z within scene bounds.
    :param seed: Random seed for reproducibility
    :param bounds: Optional custom bounds as (min_x, max_x, min_y, max_y).
        If None, uses scene bounding box.
    
    :return: Sampled positions as mi.Point3f with shape [num_samples, 3]
    """
    # Set random seed
    np.random.seed(seed)
    
    # Get scene bounds
    if bounds is None:
        bbox = scene.mi_scene.bbox()
        min_point = bbox.min
        max_point = bbox.max
        
        # Handle infinite bounds (empty scene)
        min_point = dr.select(dr.isinf(min_point), -10.0, min_point)
        max_point = dr.select(dr.isinf(max_point), 10.0, max_point)
        
        min_x = float(min_point.x)
        max_x = float(max_point.x)
        min_y = float(min_point.y)
        max_y = float(max_point.y)
        min_z = float(min_point.z)
        max_z = float(max_point.z)
    else:
        min_x, max_x, min_y, max_y = bounds
        bbox = scene.mi_scene.bbox()
        min_point = bbox.min
        max_point = bbox.max
        min_z = float(dr.select(dr.isinf(min_point.z), 0.0, min_point.z))
        max_z = float(dr.select(dr.isinf(max_point.z), 3.0, max_point.z))
    
    # Sample random positions
    x_samples = np.random.uniform(min_x, max_x, num_samples)
    y_samples = np.random.uniform(min_y, max_y, num_samples)
    
    if z_height is not None:
        z_samples = np.full(num_samples, z_height)
    else:
        z_samples = np.random.uniform(min_z, max_z, num_samples)
    
    # Convert to mitsuba Point3f
    positions = mi.Point3f(
        mi.Float(x_samples),
        mi.Float(y_samples),
        mi.Float(z_samples)
    )
    
    return positions


def compute_signal_strength_at_positions(
    scene: 'Scene',
    positions: mi.Point3f,
    max_depth: int = 3,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = False,
    refraction: bool = False,
    diffraction: bool = False,
    edge_diffraction: bool = False,
    synthetic_array: bool = True,
    samples_per_src: int = 1000000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute signal strength metrics at specified positions.
    
    This function creates temporary receivers at the specified positions,
    uses the PathSolver to compute propagation paths, and extracts
    signal strength information for each transmitter-receiver pair.
    
    :param scene: Scene containing transmitters
    :param positions: Positions at which to compute signal strength [num_samples, 3]
    :param max_depth: Maximum number of reflections/interactions
    :param los: Enable line-of-sight paths
    :param specular_reflection: Enable specular reflection
    :param diffuse_reflection: Enable diffuse reflection
    :param refraction: Enable refraction
    :param diffraction: Enable diffraction
    :param edge_diffraction: Enable edge diffraction
    :param synthetic_array: Use synthetic array modeling
    :param samples_per_src: Number of rays per source for path tracing
    
    :return: Tuple of (positions_np, path_gain, rss, sinr)
        - positions_np: numpy array of positions [num_samples, 3]
        - path_gain: numpy array of path gains [num_tx, num_samples]
        - rss: numpy array of received signal strength in dBm [num_tx, num_samples]
        - sinr: numpy array of SINR in dB [num_tx, num_samples]
    """
    # Import here to avoid circular dependency
    from sionna.rt import Receiver, PathSolver
    
    num_samples = dr.width(positions)
    num_tx = len(scene.transmitters)
    
    if num_tx == 0:
        raise ValueError("Scene must contain at least one transmitter")
    
    # Store original receivers to restore later
    original_receivers = dict(scene.receivers)
    
    # Remove all existing receivers
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)
    
    # Create temporary receivers at sampled positions
    temp_receivers = []
    for i in range(num_samples):
        pos = mi.Point3f(positions.x[i], positions.y[i], positions.z[i])
        rx_name = f"_temp_rx_{i}"
        rx = Receiver(name=rx_name, position=pos, orientation=mi.Point3f(0, 0, 0))
        scene.add(rx)
        temp_receivers.append(rx_name)
    
    try:
        # Compute paths using PathSolver
        path_solver = PathSolver()
        paths = path_solver(
            scene,
            max_depth=max_depth,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            refraction=refraction,
            diffraction=diffraction,
            edge_diffraction=edge_diffraction,
            synthetic_array=synthetic_array,
            samples_per_src=samples_per_src
        )
        
        # Extract path coefficients
        # paths.a is a tuple (real, imag)
        a_real = paths.a[0].numpy()
        a_imag = paths.a[1].numpy()
        a = a_real + 1j * a_imag
        
        # Shape depends on synthetic_array flag
        if synthetic_array:
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            # Apply transmit precoding (assuming default precoding)
            a /= np.sqrt(a.shape[3])
            a = np.sum(a, axis=3)
            
            # Sum over receive antennas and paths to get total power
            # [num_rx, num_tx]
            path_gain = np.sum(np.abs(a)**2, axis=(1, 3))
        else:
            # Without synthetic arrays
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            a /= np.sqrt(a.shape[3])
            a = np.sum(a, axis=3)
            path_gain = np.sum(np.abs(a)**2, axis=(1, 3))
        
        # Transpose to [num_tx, num_rx]
        path_gain = path_gain.T
        
        # Get transmitter powers
        tx_powers = np.array([tx.power[0] for tx in scene.transmitters.values()])
        tx_powers = tx_powers.reshape(-1, 1)  # [num_tx, 1]
        
        # Compute RSS in watts
        rss_watts = path_gain * tx_powers  # [num_tx, num_samples]
        
        # Convert RSS to dBm
        rss_dbm = 10 * np.log10(rss_watts * 1000)  # Convert W to mW
        rss_dbm = np.where(np.isfinite(rss_dbm), rss_dbm, -np.inf)
        
        # Compute SINR
        # Total received power from all transmitters
        total_power = np.sum(rss_watts, axis=0, keepdims=True)  # [1, num_samples]
        
        # Interference for each transmitter (total - own signal)
        interference = total_power - rss_watts  # [num_tx, num_samples]
        
        # Thermal noise power
        noise_power = float(scene.thermal_noise_power)
        
        # SINR in linear scale
        sinr_linear = rss_watts / (interference + noise_power)
        
        # Convert SINR to dB
        sinr_db = 10 * np.log10(sinr_linear)
        sinr_db = np.where(np.isfinite(sinr_db), sinr_db, -np.inf)
        
        # Extract positions as numpy array
        positions_np = np.column_stack([
            positions.x.numpy() if hasattr(positions.x, 'numpy') else np.array([float(positions.x[i]) for i in range(num_samples)]),
            positions.y.numpy() if hasattr(positions.y, 'numpy') else np.array([float(positions.y[i]) for i in range(num_samples)]),
            positions.z.numpy() if hasattr(positions.z, 'numpy') else np.array([float(positions.z[i]) for i in range(num_samples)])
        ])
        
    finally:
        # Clean up: remove temporary receivers
        for rx_name in temp_receivers:
            scene.remove(rx_name)
        
        # Restore original receivers
        for rx_name, rx in original_receivers.items():
            scene.add(rx)
    
    return positions_np, path_gain, rss_dbm, sinr_db


def sample_signal_data(
    scene: 'Scene',
    num_samples: int = 1000,
    z_height: float | None = 1.5,
    seed: int = 42,
    bounds: Tuple[float, float, float, float] | None = None,
    max_depth: int = 3,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = False,
    refraction: bool = False,
    diffraction: bool = False,
    edge_diffraction: bool = False,
    synthetic_array: bool = True,
    samples_per_src: int = 1000000,
    output_format: str = "dict"
) -> Dict | List[Dict]:
    """
    Sample random positions in a scene and compute signal strength data.
    
    This is the main convenience function that combines position sampling
    and signal strength computation. It returns structured data containing
    position coordinates and signal level information from all transmitters.
    
    :param scene: Scene containing transmitters and geometry
    :param num_samples: Number of random points to sample
    :param z_height: Fixed z-coordinate (height) for sampling. 
        If None, samples z randomly within scene bounds.
    :param seed: Random seed for reproducibility
    :param bounds: Optional custom sampling bounds as (min_x, max_x, min_y, max_y).
        If None, uses scene bounding box.
    :param max_depth: Maximum number of reflections/interactions for ray tracing
    :param los: Enable line-of-sight paths
    :param specular_reflection: Enable specular reflection
    :param diffuse_reflection: Enable diffuse reflection
    :param refraction: Enable refraction
    :param diffraction: Enable diffraction
    :param edge_diffraction: Enable edge diffraction
    :param synthetic_array: Use synthetic array modeling for efficiency
    :param samples_per_src: Number of rays per source for path tracing
    :param output_format: Output format: "dict" or "list"
        - "dict": Returns a dictionary with arrays for each metric
        - "list": Returns a list of dictionaries, one per sample point
    
    :return: Structured data containing:
        If output_format="dict":
            {
                'positions': numpy array [num_samples, 3],
                'tx_names': list of transmitter names,
                'path_gain': numpy array [num_tx, num_samples],
                'rss_dbm': numpy array [num_tx, num_samples],
                'sinr_db': numpy array [num_tx, num_samples]
            }
        If output_format="list":
            List of dictionaries, one per sample:
            [
                {
                    'position': (x, y, z),
                    'transmitters': {
                        'tx_name_1': {'path_gain': ..., 'rss_dbm': ..., 'sinr_db': ...},
                        'tx_name_2': {'path_gain': ..., 'rss_dbm': ..., 'sinr_db': ...},
                        ...
                    }
                },
                ...
            ]
    
    Example
    -------
    .. code-block:: python
    
        import sionna.rt as rt
        from sionna.rt.utils import sample_signal_data
        import mitsuba as mi
        
        # Load scene
        scene = rt.load_scene(rt.scene.munich)
        
        # Add transmitters
        scene.add(rt.Transmitter(name="tx1", position=mi.Point3f(10, 20, 30)))
        scene.add(rt.Transmitter(name="tx2", position=mi.Point3f(-10, 30, 25)))
        
        # Configure antenna arrays
        scene.tx_array = rt.PlanarArray(num_rows=4, num_cols=4, 
                                       pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                       pattern="iso", polarization="VH")
        
        # Sample 500 random points at height 1.5m
        data = sample_signal_data(scene, num_samples=500, z_height=1.5)
        
        # Access results
        print("Positions shape:", data['positions'].shape)
        print("RSS from tx1:", data['rss_dbm'][0])  # First transmitter
        print("SINR from tx2:", data['sinr_db'][1])  # Second transmitter
    """
    # Get transmitter names
    tx_names = list(scene.transmitters.keys())
    
    # Sample random positions
    positions = sample_random_positions(
        scene, num_samples, z_height=z_height, seed=seed, bounds=bounds
    )
    
    # Compute signal strength at positions
    positions_np, path_gain, rss_dbm, sinr_db = compute_signal_strength_at_positions(
        scene, positions,
        max_depth=max_depth,
        los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        synthetic_array=synthetic_array,
        samples_per_src=samples_per_src
    )
    
    if output_format == "dict":
        return {
            'positions': positions_np,
            'tx_names': tx_names,
            'path_gain': path_gain,
            'rss_dbm': rss_dbm,
            'sinr_db': sinr_db
        }
    elif output_format == "list":
        # Convert to list of dictionaries
        result = []
        for i in range(num_samples):
            sample_data = {
                'position': tuple(positions_np[i]),
                'transmitters': {}
            }
            for j, tx_name in enumerate(tx_names):
                sample_data['transmitters'][tx_name] = {
                    'path_gain': float(path_gain[j, i]),
                    'rss_dbm': float(rss_dbm[j, i]),
                    'sinr_db': float(sinr_db[j, i])
                }
            result.append(sample_data)
        return result
    else:
        raise ValueError("output_format must be 'dict' or 'list'")
