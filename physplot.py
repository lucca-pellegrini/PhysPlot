"""
PhysPlot - Physics Plotting Library with Regression Analysis

A simple API for creating professional scientific plots with LaTeX formatting
and regression analysis capabilities.
"""

import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


class PhysPlot:
    """
    A class for creating professional scientific plots with regression analysis.
    """

    def __init__(self):
        """Initialize the plotter with LaTeX settings for professional appearance."""
        # Configure matplotlib for LaTeX rendering and modern appearance
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern"],
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 12,
                "figure.titlesize": 18,
                "lines.linewidth": 2,
                "lines.markersize": 6,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 1.2,
                "xtick.major.width": 1.2,
                "ytick.major.width": 1.2,
                "figure.figsize": (10, 7),
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.format": "pdf",
                "savefig.bbox": "tight",
            }
        )

    def plot_with_regression(
        self,
        data: Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple, Tuple]],
        func: Callable,
        initial_guess: Optional[Tuple] = None,
        xlabel: str = "x",
        ylabel: str = "y",
        title: str = "Physics Plot with Regression",
        filename: str = "plot.pdf",
        data_label: str = "Data",
        fit_label: str = "Fit",
        show_equation: bool = True,
        param_names: Optional[Tuple[str, ...]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a plot with regression analysis.

        Parameters:
        -----------
        data : tuple of arrays or tuples
            (x_values, y_values) where each can be numpy arrays or tuples
        func : callable
            Function to fit. Should accept x as first parameter, then fit parameters
        initial_guess : tuple, optional
            Initial guess for parameters. If None, will use ones.
        xlabel : str
            X-axis label (LaTeX formatting supported)
        ylabel : str
            Y-axis label (LaTeX formatting supported)
        title : str
            Plot title (LaTeX formatting supported)
        filename : str
            Output filename (should end with .pdf)
        data_label : str
            Label for data points in legend
        fit_label : str
            Label for fit line in legend
        show_equation : bool
            Whether to show the fitted equation in the legend
        param_names : tuple of str, optional
            Names for parameters (for equation display)
        **kwargs : dict
            Additional arguments passed to curve_fit

        Returns:
        --------
        dict : Dictionary containing fit results
            - 'params': fitted parameters
            - 'r_squared': R² value
            - 'params_std': standard errors of parameters
            - 'covariance': covariance matrix
        """

        # Convert data to numpy arrays if needed
        x_data, y_data = data
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Determine number of parameters from function signature
        import inspect

        sig = inspect.signature(func)
        n_params = len(sig.parameters) - 1  # Subtract 1 for x parameter

        # Set default initial guess if not provided
        if initial_guess is None:
            initial_guess = np.ones(n_params)

        # Perform curve fitting
        try:
            popt, pcov = curve_fit(func, x_data, y_data, p0=initial_guess, **kwargs)
        except Exception as e:
            warnings.warn(f"Curve fitting failed: {e}")
            return None

        # Calculate R²
        y_pred = func(x_data, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate parameter standard errors
        param_errors = np.sqrt(np.diag(pcov))

        # Create the plot
        fig, ax = plt.subplots()

        # Plot data points
        ax.scatter(x_data, y_data, alpha=0.7, label=data_label, zorder=5)

        # Create smooth curve for fit
        x_smooth = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_smooth = func(x_smooth, *popt)

        # Plot fit line
        fit_label_with_r2 = f"{fit_label} ($R^2 = {r_squared:.4f}$)"
        ax.plot(x_smooth, y_smooth, "r-", label=fit_label_with_r2, zorder=10)

        # Add equation to legend if requested
        if show_equation and param_names:
            equation_parts = []
            for i, (param, name) in enumerate(zip(popt, param_names)):
                equation_parts.append(f"{name} = {param:.4f}")
            equation_text = ", ".join(equation_parts)
            ax.text(
                0.05,
                0.95,
                equation_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Add legend
        ax.legend(frameon=True, fancybox=True, shadow=True)

        # Tight layout and save
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        # Return results
        return {
            "params": popt,
            "r_squared": r_squared,
            "params_std": param_errors,
            "covariance": pcov,
            "function": func,
        }

    def multi_plot_comparison(
        self,
        datasets: list,
        functions: list,
        labels: list,
        xlabel: str = "x",
        ylabel: str = "y",
        title: str = "Regression Comparison",
        filename: str = "comparison.pdf",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple datasets with different regression functions.

        Parameters:
        -----------
        datasets : list of tuples
            List of (x_data, y_data) tuples
        functions : list of callables
            List of functions to fit to each dataset
        labels : list of str
            Labels for each dataset/function pair
        xlabel, ylabel, title, filename : str
            Plot formatting parameters

        Returns:
        --------
        dict : Dictionary with results for each dataset
        """

        fig, ax = plt.subplots()
        results = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

        for i, (data, func, label) in enumerate(zip(datasets, functions, labels)):
            x_data, y_data = data
            x_data, y_data = np.array(x_data), np.array(y_data)

            # Fit the function
            try:
                popt, pcov = curve_fit(func, x_data, y_data)

                # Calculate R²
                y_pred = func(x_data, *popt)
                ss_res = np.sum((y_data - y_pred) ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

                # Plot
                ax.scatter(
                    x_data, y_data, alpha=0.7, color=colors[i], label=f"{label} Data"
                )

                x_smooth = np.linspace(np.min(x_data), np.max(x_data), 200)
                y_smooth = func(x_smooth, *popt)
                ax.plot(
                    x_smooth,
                    y_smooth,
                    "--",
                    color=colors[i],
                    label=f"{label} Fit ($R^2 = {r_squared:.3f}$)",
                )

                results[label] = {
                    "params": popt,
                    "r_squared": r_squared,
                    "params_std": np.sqrt(np.diag(pcov)),
                    "covariance": pcov,
                }

            except Exception as e:
                warnings.warn(f"Fitting failed for {label}: {e}")
                results[label] = None

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        return results


def linear(x, a, b):
    """Linear function: y = ax + b"""
    return a * x + b


def quadratic(x, a, b, c):
    """Quadratic function: y = ax² + bx + c"""
    return a * x**2 + b * x + c


def cubic(x, a, b, c, d):
    """Cubic function: y = ax³ + bx² + cx + d"""
    return a * x**3 + b * x**2 + c * x + d


def exponential(x, a, b, c):
    """Exponential function: y = a * exp(bx) + c"""
    return a * np.exp(b * x) + c


def power_law(x, a, b):
    """Power law function: y = ax^b"""
    return a * np.power(x, b)


def gaussian(x, a, mu, sigma):
    """Gaussian function: y = a * exp(-(x-mu)²/(2σ²))"""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))