#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import numpy as np
import mitsuba as mi
import drjit as dr
import sionna.rt as rt
from sionna.rt.utils import (
    sample_random_positions,
    compute_signal_strength_at_positions,
    sample_signal_data
)


class TestSampling:
    """Test cases for signal strength sampling utilities"""
    
    def test_sample_random_positions_basic(self):
        """Test basic position sampling"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        num_samples = 10
        positions = sample_random_positions(scene, num_samples, z_height=1.5, seed=42)
        
        # Check that we get the correct number of samples
        assert dr.width(positions) == num_samples
        
        # Check that z-coordinates are all 1.5
        for i in range(num_samples):
            assert abs(float(positions.z[i]) - 1.5) < 1e-6
    
    def test_sample_random_positions_with_bounds(self):
        """Test position sampling with custom bounds"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        num_samples = 20
        bounds = (-10.0, 10.0, -5.0, 5.0)  # min_x, max_x, min_y, max_y
        positions = sample_random_positions(
            scene, num_samples, z_height=2.0, seed=123, bounds=bounds
        )
        
        # Check bounds
        for i in range(num_samples):
            x = float(positions.x[i])
            y = float(positions.y[i])
            z = float(positions.z[i])
            
            assert bounds[0] <= x <= bounds[1], f"x={x} out of bounds"
            assert bounds[2] <= y <= bounds[3], f"y={y} out of bounds"
            assert abs(z - 2.0) < 1e-6, f"z={z} should be 2.0"
    
    def test_sample_random_positions_random_z(self):
        """Test position sampling with random z coordinate"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        num_samples = 10
        positions = sample_random_positions(scene, num_samples, z_height=None, seed=42)
        
        # Check that we get the correct number of samples
        assert dr.width(positions) == num_samples
        
        # Check that z-coordinates vary (not all the same)
        z_values = [float(positions.z[i]) for i in range(num_samples)]
        assert len(set(z_values)) > 1, "Z values should vary when z_height is None"
    
    def test_compute_signal_strength_single_tx(self):
        """Test signal strength computation with a single transmitter"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        # Configure arrays
        scene.tx_array = rt.PlanarArray(num_rows=2, num_cols=2, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        
        # Add transmitter
        tx = rt.Transmitter(name="tx1", position=mi.Point3f(0, 0, 10))
        scene.add(tx)
        
        # Sample positions
        num_samples = 5
        positions = sample_random_positions(scene, num_samples, z_height=1.5, seed=42)
        
        # Compute signal strength
        positions_np, path_gain, rss_dbm, sinr_db = compute_signal_strength_at_positions(
            scene, positions, max_depth=2, los=True, specular_reflection=True
        )
        
        # Check shapes
        assert positions_np.shape == (num_samples, 3)
        assert path_gain.shape == (1, num_samples)  # 1 transmitter
        assert rss_dbm.shape == (1, num_samples)
        assert sinr_db.shape == (1, num_samples)
        
        # Check that values are reasonable (non-NaN where there's signal)
        assert not np.all(np.isnan(path_gain))
    
    def test_compute_signal_strength_multiple_tx(self):
        """Test signal strength computation with multiple transmitters"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        # Configure arrays
        scene.tx_array = rt.PlanarArray(num_rows=2, num_cols=2, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        
        # Add multiple transmitters
        tx1 = rt.Transmitter(name="tx1", position=mi.Point3f(-20, 0, 10))
        tx2 = rt.Transmitter(name="tx2", position=mi.Point3f(20, 0, 10))
        scene.add(tx1)
        scene.add(tx2)
        
        # Sample positions
        num_samples = 5
        positions = sample_random_positions(scene, num_samples, z_height=1.5, seed=42)
        
        # Compute signal strength
        positions_np, path_gain, rss_dbm, sinr_db = compute_signal_strength_at_positions(
            scene, positions, max_depth=2, los=True, specular_reflection=True
        )
        
        # Check shapes for 2 transmitters
        assert positions_np.shape == (num_samples, 3)
        assert path_gain.shape == (2, num_samples)  # 2 transmitters
        assert rss_dbm.shape == (2, num_samples)
        assert sinr_db.shape == (2, num_samples)
    
    def test_sample_signal_data_dict_format(self):
        """Test sample_signal_data with dict output format"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        # Configure arrays
        scene.tx_array = rt.PlanarArray(num_rows=2, num_cols=2, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        
        # Add transmitter
        tx = rt.Transmitter(name="tx1", position=mi.Point3f(0, 0, 10))
        scene.add(tx)
        
        # Sample signal data
        num_samples = 10
        data = sample_signal_data(
            scene,
            num_samples=num_samples,
            z_height=1.5,
            seed=42,
            max_depth=2,
            output_format="dict"
        )
        
        # Check output structure
        assert 'positions' in data
        assert 'tx_names' in data
        assert 'path_gain' in data
        assert 'rss_dbm' in data
        assert 'sinr_db' in data
        
        # Check data types and shapes
        assert isinstance(data['positions'], np.ndarray)
        assert isinstance(data['tx_names'], list)
        assert isinstance(data['path_gain'], np.ndarray)
        assert isinstance(data['rss_dbm'], np.ndarray)
        assert isinstance(data['sinr_db'], np.ndarray)
        
        assert data['positions'].shape == (num_samples, 3)
        assert len(data['tx_names']) == 1
        assert data['path_gain'].shape == (1, num_samples)
    
    def test_sample_signal_data_list_format(self):
        """Test sample_signal_data with list output format"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        
        # Configure arrays
        scene.tx_array = rt.PlanarArray(num_rows=2, num_cols=2, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        
        # Add transmitters
        tx1 = rt.Transmitter(name="tx1", position=mi.Point3f(-10, 0, 10))
        tx2 = rt.Transmitter(name="tx2", position=mi.Point3f(10, 0, 10))
        scene.add(tx1)
        scene.add(tx2)
        
        # Sample signal data
        num_samples = 5
        data = sample_signal_data(
            scene,
            num_samples=num_samples,
            z_height=1.5,
            seed=42,
            max_depth=2,
            output_format="list"
        )
        
        # Check output structure
        assert isinstance(data, list)
        assert len(data) == num_samples
        
        # Check structure of each sample
        for sample in data:
            assert 'position' in sample
            assert 'transmitters' in sample
            assert isinstance(sample['position'], tuple)
            assert len(sample['position']) == 3
            assert isinstance(sample['transmitters'], dict)
            
            # Check that we have data for both transmitters
            assert 'tx1' in sample['transmitters']
            assert 'tx2' in sample['transmitters']
            
            # Check metrics for each transmitter
            for tx_data in sample['transmitters'].values():
                assert 'path_gain' in tx_data
                assert 'rss_dbm' in tx_data
                assert 'sinr_db' in tx_data
    
    def test_sample_signal_data_invalid_format(self):
        """Test that invalid output format raises an error"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        scene.add(rt.Transmitter(name="tx1", position=mi.Point3f(0, 0, 10)))
        
        with pytest.raises(ValueError, match="output_format must be 'dict' or 'list'"):
            sample_signal_data(
                scene,
                num_samples=5,
                output_format="invalid"
            )
    
    def test_sample_signal_data_no_transmitters(self):
        """Test that sampling without transmitters raises an error"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        
        # Don't add any transmitters
        positions = sample_random_positions(scene, 5, z_height=1.5)
        
        with pytest.raises(ValueError, match="Scene must contain at least one transmitter"):
            compute_signal_strength_at_positions(scene, positions)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        scene = rt.load_scene(rt.scene.simple_street_canyon)
        scene.tx_array = rt.PlanarArray(num_rows=2, num_cols=2, pattern="iso", polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        scene.add(rt.Transmitter(name="tx1", position=mi.Point3f(0, 0, 10)))
        
        # Sample twice with same seed
        data1 = sample_signal_data(scene, num_samples=10, seed=999, output_format="dict")
        data2 = sample_signal_data(scene, num_samples=10, seed=999, output_format="dict")
        
        # Positions should be identical
        np.testing.assert_array_equal(data1['positions'], data2['positions'])
        
        # Signal metrics should be very close (allowing for minor numerical differences)
        np.testing.assert_allclose(data1['path_gain'], data2['path_gain'], rtol=1e-5, atol=1e-15)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
