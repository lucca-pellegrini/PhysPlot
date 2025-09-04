#!/usr/bin/env python3
"""
Simple example without LaTeX dependencies for quick testing
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

from physplot import PhysPlot, quadratic

# Override LaTeX settings for compatibility
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
    }
)


def main():
    # Initialize plotter
    plotter = PhysPlot()

    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 30)
    y = quadratic(x, 0.5, -2, 5) + np.random.normal(0, 0.8, len(x))

    # Perform regression
    results = plotter.plot_with_regression(
        data=(x, y),
        func=quadratic,
        xlabel="x [units]",
        ylabel="y [units]",
        title="Simple Quadratic Regression Example",
        filename="simple_example.pdf",
        param_names=("a", "b", "c"),
    )

    print("Simple example completed!")
    print(f"Fitted parameters:")
    print(f"  a = {results['params'][0]:.3f} ± {results['params_std'][0]:.3f}")
    print(f"  b = {results['params'][1]:.3f} ± {results['params_std'][1]:.3f}")
    print(f"  c = {results['params'][2]:.3f} ± {results['params_std'][2]:.3f}")
    print(f"R² = {results['r_squared']:.4f}")


if __name__ == "__main__":
    main()
