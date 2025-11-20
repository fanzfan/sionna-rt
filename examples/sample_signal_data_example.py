#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Example script demonstrating how to sample random points and get signal strength data.

This example shows how to:
1. Load a scene and add transmitters (base stations)
2. Sample random points in the scene
3. Compute signal strength (path gain, RSS, SINR) at those points
4. Export the data for analysis
"""

import sionna.rt as rt
from sionna.rt.utils import sample_signal_data
import mitsuba as mi
import numpy as np
import json


def main():
    print("=" * 70)
    print("Signal Strength Sampling Example")
    print("=" * 70)
    
    # Load a scene
    print("\n1. Loading scene...")
    scene = rt.load_scene(rt.scene.simple_street_canyon)
    
    # Configure frequency
    scene.frequency = 3.5e9  # 3.5 GHz
    
    # Configure antenna arrays for transmitters and receivers
    print("2. Configuring antenna arrays...")
    scene.tx_array = rt.PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V"
    )
    
    scene.rx_array = rt.PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="VH"
    )
    
    # Add transmitters (base stations)
    print("3. Adding transmitters (base stations)...")
    
    # Base station 1
    tx1 = rt.Transmitter(
        name="bs1",
        position=mi.Point3f(-20, 0, 10),
        orientation=mi.Point3f(0, 0, 0)
    )
    scene.add(tx1)
    
    # Base station 2
    tx2 = rt.Transmitter(
        name="bs2",
        position=mi.Point3f(20, 0, 10),
        orientation=mi.Point3f(0, 0, 0)
    )
    scene.add(tx2)
    
    print(f"   Added {len(scene.transmitters)} transmitters")
    
    # Sample random positions and compute signal strength
    print("\n4. Sampling random positions and computing signal strength...")
    print("   This may take a moment...")
    
    num_samples = 100  # Sample 100 random points
    
    # Sample points at a fixed height of 1.5m (typical mobile device height)
    data = sample_signal_data(
        scene,
        num_samples=num_samples,
        z_height=1.5,  # Fixed height at 1.5m
        seed=42,
        max_depth=3,  # Allow up to 3 reflections
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        synthetic_array=True,
        output_format="dict"  # Get structured dictionary output
    )
    
    print(f"   Sampled {num_samples} points")
    
    # Display results
    print("\n5. Results Summary:")
    print("=" * 70)
    
    positions = data['positions']
    tx_names = data['tx_names']
    path_gain = data['path_gain']
    rss_dbm = data['rss_dbm']
    sinr_db = data['sinr_db']
    
    print(f"\nPositions shape: {positions.shape}")
    print(f"Number of transmitters: {len(tx_names)}")
    print(f"Transmitter names: {tx_names}")
    
    # Statistics for each transmitter
    for i, tx_name in enumerate(tx_names):
        print(f"\n--- Transmitter: {tx_name} ---")
        print(f"  Path Gain (linear):")
        print(f"    Mean: {np.mean(path_gain[i]):.6f}")
        print(f"    Min:  {np.min(path_gain[i]):.6f}")
        print(f"    Max:  {np.max(path_gain[i]):.6f}")
        
        # Filter out -inf values for statistics
        valid_rss = rss_dbm[i][np.isfinite(rss_dbm[i])]
        if len(valid_rss) > 0:
            print(f"  RSS (dBm):")
            print(f"    Mean: {np.mean(valid_rss):.2f} dBm")
            print(f"    Min:  {np.min(valid_rss):.2f} dBm")
            print(f"    Max:  {np.max(valid_rss):.2f} dBm")
        
        valid_sinr = sinr_db[i][np.isfinite(sinr_db[i])]
        if len(valid_sinr) > 0:
            print(f"  SINR (dB):")
            print(f"    Mean: {np.mean(valid_sinr):.2f} dB")
            print(f"    Min:  {np.min(valid_sinr):.2f} dB")
            print(f"    Max:  {np.max(valid_sinr):.2f} dB")
    
    # Show some sample data points
    print("\n6. Sample Data Points (first 5):")
    print("=" * 70)
    
    for i in range(min(5, num_samples)):
        print(f"\nPoint {i+1}:")
        print(f"  Position: ({positions[i,0]:.2f}, {positions[i,1]:.2f}, {positions[i,2]:.2f})")
        for j, tx_name in enumerate(tx_names):
            print(f"  {tx_name}:")
            print(f"    Path Gain: {path_gain[j,i]:.6f}")
            print(f"    RSS: {rss_dbm[j,i]:.2f} dBm")
            print(f"    SINR: {sinr_db[j,i]:.2f} dB")
    
    # Alternative: Use list output format
    print("\n7. Alternative list format example:")
    print("=" * 70)
    
    data_list = sample_signal_data(
        scene,
        num_samples=5,  # Just 5 samples for demonstration
        z_height=1.5,
        seed=100,
        output_format="list"
    )
    
    print(f"\nGot {len(data_list)} samples in list format")
    print("\nFirst sample:")
    print(json.dumps(data_list[0], indent=2, default=str))
    
    # Export to JSON file (optional)
    print("\n8. Exporting data to JSON file...")
    output_file = "/tmp/signal_strength_data.json"
    
    # Convert numpy arrays to lists for JSON serialization
    export_data = {
        'positions': positions.tolist(),
        'tx_names': tx_names,
        'path_gain': path_gain.tolist(),
        'rss_dbm': rss_dbm.tolist(),
        'sinr_db': sinr_db.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"   Data exported to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
