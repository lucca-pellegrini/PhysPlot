#!/usr/bin/env python3
"""
Example usage of the Scientific Plotter API

This script demonstrates how to use the ScientificPlotter class for
regression analysis and professional plotting with various functions.
"""

import numpy as np

from scientific_plotter import (
    ScientificPlotter,
    exponential,
    gaussian,
    linear,
    quadratic,
)


def custom_function(x, a, b, c):
    """Custom function example: y = a*sin(bx) + c"""
    return a * np.sin(b * x) + c


def main():
    # Initialize the plotter
    plotter = ScientificPlotter()

    # Set random seed for reproducible results
    np.random.seed(42)

    # Example 1: Linear regression
    print("Example 1: Linear Regression")
    print("-" * 40)

    # Generate synthetic linear data with noise
    x_linear = np.linspace(0, 10, 50)
    true_a, true_b = 2.5, 1.3
    y_linear = linear(x_linear, true_a, true_b) + np.random.normal(
        0, 0.5, len(x_linear)
    )

    # Perform regression
    results_linear = plotter.plot_with_regression(
        data=(x_linear, y_linear),
        func=linear,
        xlabel=r"$x$ [units]",
        ylabel=r"$y$ [units]",
        title=r"Linear Regression Example: $y = ax + b$",
        filename="linear_regression.pdf",
        param_names=("a", "b"),
        data_label="Experimental Data",
        fit_label="Linear Fit",
    )

    print(f"True parameters: a = {true_a}, b = {true_b}")
    print(
        f"Fitted parameters: a = {results_linear['params'][0]:.3f} ± {results_linear['params_std'][0]:.3f}"
    )
    print(
        f"                   b = {results_linear['params'][1]:.3f} ± {results_linear['params_std'][1]:.3f}"
    )
    print(f"R² = {results_linear['r_squared']:.4f}")
    print()

    # Example 2: Quadratic regression
    print("Example 2: Quadratic Regression")
    print("-" * 40)

    # Generate synthetic quadratic data
    x_quad = np.linspace(-3, 3, 60)
    true_a, true_b, true_c = 0.8, -1.2, 2.0
    y_quad = quadratic(x_quad, true_a, true_b, true_c) + np.random.normal(
        0, 0.3, len(x_quad)
    )

    results_quad = plotter.plot_with_regression(
        data=(x_quad, y_quad),
        func=quadratic,
        xlabel=r"$x$ [m]",
        ylabel=r"$y$ [m$^2$]",
        title=r"Quadratic Regression: $y = ax^2 + bx + c$",
        filename="quadratic_regression.pdf",
        param_names=("a", "b", "c"),
        data_label="Measured Points",
        fit_label="Quadratic Fit",
    )

    print(f"True parameters: a = {true_a}, b = {true_b}, c = {true_c}")
    print(
        f"Fitted parameters: a = {results_quad['params'][0]:.3f} ± {results_quad['params_std'][0]:.3f}"
    )
    print(
        f"                   b = {results_quad['params'][1]:.3f} ± {results_quad['params_std'][1]:.3f}"
    )
    print(
        f"                   c = {results_quad['params'][2]:.3f} ± {results_quad['params_std'][2]:.3f}"
    )
    print(f"R² = {results_quad['r_squared']:.4f}")
    print()

    # Example 3: Exponential decay
    print("Example 3: Exponential Decay")
    print("-" * 40)

    x_exp = np.linspace(0, 5, 40)
    true_a, true_b, true_c = 10.0, -0.8, 1.0
    y_exp = exponential(x_exp, true_a, true_b, true_c) + np.random.normal(
        0, 0.4, len(x_exp)
    )

    results_exp = plotter.plot_with_regression(
        data=(x_exp, y_exp),
        func=exponential,
        initial_guess=(10, -1, 1),  # Good initial guess for exponential
        xlabel=r"Time $t$ [s]",
        ylabel=r"Amplitude $A$ [V]",
        title=r"Exponential Decay: $A = A_0 e^{-\lambda t} + A_{\infty}$",
        filename="exponential_decay.pdf",
        param_names=(r"A_0", r"\lambda", r"A_{\infty}"),
        data_label="Measurements",
        fit_label="Exponential Fit",
    )

    print(f"True parameters: A₀ = {true_a}, λ = {true_b}, A∞ = {true_c}")
    print(
        f"Fitted parameters: A₀ = {results_exp['params'][0]:.3f} ± {results_exp['params_std'][0]:.3f}"
    )
    print(
        f"                   λ = {results_exp['params'][1]:.3f} ± {results_exp['params_std'][1]:.3f}"
    )
    print(
        f"                   A∞ = {results_exp['params'][2]:.3f} ± {results_exp['params_std'][2]:.3f}"
    )
    print(f"R² = {results_exp['r_squared']:.4f}")
    print()

    # Example 4: Gaussian distribution
    print("Example 4: Gaussian Distribution")
    print("-" * 40)

    x_gauss = np.linspace(-5, 5, 80)
    true_a, true_mu, true_sigma = 3.5, 0.5, 1.2
    y_gauss = gaussian(x_gauss, true_a, true_mu, true_sigma) + np.random.normal(
        0, 0.1, len(x_gauss)
    )

    results_gauss = plotter.plot_with_regression(
        data=(x_gauss, y_gauss),
        func=gaussian,
        initial_guess=(3, 0, 1),
        xlabel=r"Position $x$ [cm]",
        ylabel=r"Intensity $I$ [a.u.]",
        title=r"Gaussian Profile: $I = I_0 \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$",
        filename="gaussian_fit.pdf",
        param_names=(r"I_0", r"\mu", r"\sigma"),
        data_label="Data Points",
        fit_label="Gaussian Fit",
    )

    print(f"True parameters: I₀ = {true_a}, μ = {true_mu}, σ = {true_sigma}")
    print(
        f"Fitted parameters: I₀ = {results_gauss['params'][0]:.3f} ± {results_gauss['params_std'][0]:.3f}"
    )
    print(
        f"                   μ = {results_gauss['params'][1]:.3f} ± {results_gauss['params_std'][1]:.3f}"
    )
    print(
        f"                   σ = {results_gauss['params'][2]:.3f} ± {results_gauss['params_std'][2]:.3f}"
    )
    print(f"R² = {results_gauss['r_squared']:.4f}")
    print()

    # Example 5: Custom function (sinusoidal)
    print("Example 5: Custom Sinusoidal Function")
    print("-" * 40)

    x_custom = np.linspace(0, 4 * np.pi, 100)
    true_a, true_b, true_c = 2.0, 1.5, 0.3
    y_custom = custom_function(x_custom, true_a, true_b, true_c) + np.random.normal(
        0, 0.2, len(x_custom)
    )

    results_custom = plotter.plot_with_regression(
        data=(x_custom, y_custom),
        func=custom_function,
        initial_guess=(2, 1, 0),
        xlabel=r"Angle $\theta$ [rad]",
        ylabel=r"Signal $S$ [V]",
        title=r"Sinusoidal Function: $S = A \sin(B\theta) + C$",
        filename="custom_sinusoidal.pdf",
        param_names=("A", "B", "C"),
        data_label="Signal Data",
        fit_label="Sinusoidal Fit",
    )

    print(f"True parameters: A = {true_a}, B = {true_b}, C = {true_c}")
    print(
        f"Fitted parameters: A = {results_custom['params'][0]:.3f} ± {results_custom['params_std'][0]:.3f}"
    )
    print(
        f"                   B = {results_custom['params'][1]:.3f} ± {results_custom['params_std'][1]:.3f}"
    )
    print(
        f"                   C = {results_custom['params'][2]:.3f} ± {results_custom['params_std'][2]:.3f}"
    )
    print(f"R² = {results_custom['r_squared']:.4f}")
    print()

    # Example 6: Multiple dataset comparison
    print("Example 6: Multiple Dataset Comparison")
    print("-" * 40)

    # Generate different datasets
    x_comp = np.linspace(0, 3, 30)

    # Linear dataset
    y_comp1 = linear(x_comp, 2, 1) + np.random.normal(0, 0.3, len(x_comp))

    # Quadratic dataset
    y_comp2 = quadratic(x_comp, 0.5, 1, 0.5) + np.random.normal(0, 0.2, len(x_comp))

    # Exponential dataset
    y_comp3 = exponential(x_comp, 3, 0.8, 0.5) + np.random.normal(0, 0.2, len(x_comp))

    comparison_results = plotter.multi_plot_comparison(
        datasets=[(x_comp, y_comp1), (x_comp, y_comp2), (x_comp, y_comp3)],
        functions=[linear, quadratic, exponential],
        labels=["Linear", "Quadratic", "Exponential"],
        xlabel=r"$x$ [units]",
        ylabel=r"$y$ [units]",
        title="Comparison of Different Regression Models",
        filename="regression_comparison.pdf",
    )

    for model, result in comparison_results.items():
        if result is not None:
            print(f"{model} Model: R² = {result['r_squared']:.4f}")

    print("\nAll plots have been saved as PDF files!")
    print("Generated files:")
    print("- linear_regression.pdf")
    print("- quadratic_regression.pdf")
    print("- exponential_decay.pdf")
    print("- gaussian_fit.pdf")
    print("- custom_sinusoidal.pdf")
    print("- regression_comparison.pdf")


if __name__ == "__main__":
    main()
