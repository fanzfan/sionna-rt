#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import drjit as dr
import mitsuba as mi

import sionna.rt as rt
from sionna.rt.radio_materials.itu import itu_material
from sionna.rt.scene import Scene, load_scene
from sionna.rt import PathSolver


def add_example_radio_devices(scene: Scene):
    # Note: hardcoded for `box_two_screens.xml` as an example.
    scene.add(rt.Transmitter("tr-1", position=[-3.0, 0.0, 1.5]))
    scene.add(rt.Receiver("rc-1", position=[3.0, 0.0, 1.5]))
    scene.add(
        rt.Receiver(
            "rc-2", position=[1.0, -2.0, 3.5], color=(0.9, 0.9, 0.2), display_radius=0.9
        )
    )

    scene.rx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )
    scene.tx_array = rt.PlanarArray(
        num_rows=1, num_cols=1, pattern="tr38901", polarization="VH"
    )


def get_example_paths(scene: Scene):
    # Ray tracing parameters
    num_samples_per_src = int(1e6)
    max_num_paths = int(1e7)
    max_depth = 3

    solver = PathSolver()
    paths = solver(
        scene,
        max_depth=max_depth,
        max_num_paths_per_src=max_num_paths,
        samples_per_src=num_samples_per_src,
    )

    return paths


def test01_preview_with_paths():
    scene = load_scene(rt.scene.box_two_screens)

    eta_r, sigma = itu_material("metal", 3e9)  # ITU material evaluated at 3GHz
    for sh in scene.mi_scene.shapes():
        material = sh.bsdf()
        material.relative_permittivity = eta_r
        material.conductivity = sigma
        material.scattering_coefficient = 0.01
        material.xpd_coefficient = 0.2

    add_example_radio_devices(scene)
    paths = get_example_paths(scene)

    scene.preview(paths=paths)


def test02_preview_with_paths_but_no_valid_path():
    scene = load_scene(rt.scene.box_two_screens)

    eta_r, sigma = itu_material("metal", 3e9)  # ITU material evaluated at 3GHz
    for sh in scene.mi_scene.shapes():
        material = sh.bsdf()
        material.relative_permittivity = eta_r
        material.conductivity = sigma
        material.scattering_coefficient = 0.01
        material.xpd_coefficient = 0.2

    add_example_radio_devices(scene)
    paths = get_example_paths(scene)

    # No valid path
    paths._valid = dr.zeros(mi.TensorXb, paths.valid.shape)

    # Should not raise ValueError: need at least one array to concatenate
    scene.preview(paths=paths)
