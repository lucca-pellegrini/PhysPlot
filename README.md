# Scientific Plotter

A Python library for creating professional scientific plots with regression analysis and LaTeX formatting.

## Features

- **Professional LaTeX formatting** for publication-ready plots
- **Flexible regression analysis** with any user-defined function
- **Built-in common functions** (linear, quadratic, exponential, Gaussian, etc.)
- **Automatic R² calculation** and parameter uncertainty estimation
- **Multiple dataset comparison** capabilities
- **PDF output** for high-quality plots

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: LaTeX must be installed on your system for proper text rendering. On Ubuntu/Debian:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

## Quick Start

```python
from scientific_plotter import ScientificPlotter, quadratic
import numpy as np

# Initialize plotter
plotter = ScientificPlotter()

# Generate sample data
x = np.linspace(0, 10, 50)
y = quadratic(x, 2, -1, 1) + np.random.normal(0, 0.5, len(x))

# Perform regression and create plot
results = plotter.plot_with_regression(
    data=(x, y),
    func=quadratic,
    xlabel=r"$x$ [units]",
    ylabel=r"$y$ [units]",
    title=r"Quadratic Fit: $y = ax^2 + bx + c$",
    filename="my_plot.pdf",
    param_names=("a", "b", "c")
)

print(f"R² = {results['r_squared']:.4f}")
print(f"Parameters: {results['params']}")
```

## API Reference

### ScientificPlotter Class

#### `plot_with_regression(data, func, **kwargs)`

Create a plot with regression analysis.

**Parameters:**
- `data`: Tuple of (x_values, y_values) 
- `func`: Function to fit (must accept x as first parameter, then fit parameters)
- `xlabel`, `ylabel`, `title`: Plot labels (LaTeX supported)
- `filename`: Output PDF filename
- `param_names`: Tuple of parameter names for equation display
- `initial_guess`: Initial parameter values for fitting

**Returns:**
Dictionary containing:
- `params`: Fitted parameters
- `r_squared`: R² value
- `params_std`: Parameter uncertainties  
- `covariance`: Covariance matrix

### Built-in Functions

- `linear(x, a, b)`: y = ax + b
- `quadratic(x, a, b, c)`: y = ax² + bx + c  
- `cubic(x, a, b, c, d)`: y = ax³ + bx² + cx + d
- `exponential(x, a, b, c)`: y = a·exp(bx) + c
- `power_law(x, a, b)`: y = ax^b
- `gaussian(x, a, mu, sigma)`: y = a·exp(-(x-μ)²/(2σ²))

## Examples

Run the example script to see the library in action:

```bash
python example.py
```

This generates several example plots demonstrating different regression types:
- Linear regression
- Quadratic fitting
- Exponential decay
- Gaussian distribution
- Custom sinusoidal function
- Multiple model comparison

## Custom Functions

Define your own functions for regression:

```python
def my_function(x, a, b, c):
    """Custom function: y = a*sin(bx) + c"""
    return a * np.sin(b * x) + c

# Use with the plotter
results = plotter.plot_with_regression(
    data=(x_data, y_data),
    func=my_function,
    initial_guess=(1, 1, 0),  # Good initial guess often helps
    param_names=("A", "f", "offset")
)
```

## LaTeX Formatting

The plotter supports full LaTeX formatting in labels and titles:

```python
xlabel=r"Temperature $T$ [K]"
ylabel=r"Resistance $R$ [$\Omega$]" 
title=r"Arrhenius Plot: $R = R_0 \exp\left(\frac{E_a}{k_B T}\right)$"
```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib  
- SciPy
- LaTeX distribution (for text rendering)