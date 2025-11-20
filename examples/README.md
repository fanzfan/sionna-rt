# Sionna RT Examples

This directory contains example scripts demonstrating how to use Sionna RT features.

## Signal Strength Sampling Example

The `sample_signal_data_example.py` script demonstrates how to:

1. Load a scene and configure base stations (transmitters)
2. Sample random points in the scene (hundreds to thousands)
3. Compute signal strength metrics at those points
4. Export the data for further analysis

### Usage

```bash
python examples/sample_signal_data_example.py
```

### Key Features

The example shows how to use the `sample_signal_data()` function to:

- **Sample random positions**: Generate random sampling points within scene bounds
- **Compute signal strength**: Calculate path gain, RSS (Received Signal Strength), and SINR (Signal-to-Interference-plus-Noise Ratio) for all transmitters
- **Export data**: Save results in JSON format for further analysis

### Output Format

The sampling function supports two output formats:

1. **Dictionary format** (`output_format="dict"`):
   ```python
   {
       'positions': numpy array [num_samples, 3],
       'tx_names': list of transmitter names,
       'path_gain': numpy array [num_tx, num_samples],
       'rss_dbm': numpy array [num_tx, num_samples],
       'sinr_db': numpy array [num_tx, num_samples]
   }
   ```

2. **List format** (`output_format="list"`):
   ```python
   [
       {
           'position': (x, y, z),
           'transmitters': {
               'tx1': {'path_gain': ..., 'rss_dbm': ..., 'sinr_db': ...},
               'tx2': {'path_gain': ..., 'rss_dbm': ..., 'sinr_db': ...}
           }
       },
       ...
   ]
   ```

### Customization

You can customize various parameters:

```python
data = sample_signal_data(
    scene,
    num_samples=1000,        # Number of points to sample
    z_height=1.5,            # Fixed height (or None for random)
    seed=42,                 # Random seed for reproducibility
    bounds=None,             # Custom bounds (min_x, max_x, min_y, max_y)
    max_depth=3,             # Maximum reflections
    los=True,                # Line-of-sight paths
    specular_reflection=True,
    diffuse_reflection=False,
    output_format="dict"     # or "list"
)
```

### Use Cases

This functionality is useful for:

- **Coverage analysis**: Assess signal coverage across a geographical area
- **Network planning**: Optimize base station placement
- **Dataset generation**: Create training data for machine learning models
- **Performance evaluation**: Analyze signal strength distribution

## More Examples

More examples will be added to this directory in the future.
