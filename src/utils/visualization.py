#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for the Continual Learning System.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_performance(performance_matrix: np.ndarray, task_names: List[str] = None):
    """
    Plot the performance of a model across sequential tasks.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training progress
                                        (task_i trained) and columns represent 
                                        performance on each task
        task_names (list): Names of the tasks
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create task names if not provided
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(performance_matrix.shape[1])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot matrix as heatmap
    sns.heatmap(performance_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
               xticklabels=task_names, yticklabels=[f"After Task {i+1}" for i in range(performance_matrix.shape[0])],
               cbar_kws={'label': 'Accuracy (%)'})
    
    # Add labels and title
    plt.xlabel("Evaluated Task")
    plt.ylabel("Training Progress")
    plt.title("Model Performance After Each Task")
    
    plt.tight_layout()
    
    return fig


def plot_forgetting(forgetting_matrix: np.ndarray, task_names: List[str] = None):
    """
    Plot the forgetting of a model across sequential tasks.
    
    Args:
        forgetting_matrix (np.ndarray): Matrix where rows represent training progress
                                       (task_i trained) and columns represent 
                                       forgetting on each task
        task_names (list): Names of the tasks
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create task names if not provided
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(forgetting_matrix.shape[1])]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot matrix as heatmap with a different colormap for forgetting
    sns.heatmap(forgetting_matrix, annot=True, fmt=".1f", cmap="Reds",
               xticklabels=task_names, yticklabels=[f"After Task {i+1}" for i in range(forgetting_matrix.shape[0])],
               cbar_kws={'label': 'Forgetting (%)'})
    
    # Add labels and title
    plt.xlabel("Task Forgotten")
    plt.ylabel("Training Progress")
    plt.title("Forgetting After Each Task")
    
    plt.tight_layout()
    
    return fig


def plot_accuracy_over_time(accuracies: List[float], task_boundaries: List[int], 
                          task_names: List[str] = None, title: str = "Accuracy Over Time"):
    """
    Plot the accuracy of a model throughout training across multiple tasks.
    
    Args:
        accuracies (list): Validation accuracies throughout training
        task_boundaries (list): Epoch indices where tasks change
        task_names (list): Names of the tasks
        title (str): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create task names if not provided
    num_tasks = len(task_boundaries) + 1
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot accuracy
    epochs = np.arange(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, 'b-', linewidth=2)
    
    # Add task boundary lines
    for boundary in task_boundaries:
        plt.axvline(x=boundary, color='r', linestyle='--')
    
    # Add task labels
    task_midpoints = [0] + task_boundaries
    task_midpoints.append(len(accuracies))
    for i in range(num_tasks):
        midpoint = (task_midpoints[i] + task_midpoints[i+1]) / 2
        plt.text(midpoint, min(accuracies) - 5, task_names[i], 
                horizontalalignment='center', fontsize=12)
    
    # Add labels and title
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.ylim(min(accuracies) - 10, max(accuracies) + 5)
    
    plt.tight_layout()
    
    return fig


def plot_task_comparison(final_accuracies: Dict[str, List[float]], task_names: List[str] = None):
    """
    Plot a comparison of final accuracies across different methods.
    
    Args:
        final_accuracies (dict): Dictionary mapping method names to lists of final accuracies on each task
        task_names (list): Names of the tasks
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get number of tasks
    num_tasks = len(next(iter(final_accuracies.values())))
    
    # Create task names if not provided
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Prepare data for grouped bar chart
    methods = list(final_accuracies.keys())
    x = np.arange(len(task_names))
    width = 0.8 / len(methods)
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        plt.bar(x + offset, final_accuracies[method], width, label=method)
    
    # Add labels and title
    plt.xlabel("Task")
    plt.ylabel("Final Accuracy (%)")
    plt.title("Comparison of Methods Across Tasks")
    plt.xticks(x, task_names)
    plt.legend()
    
    plt.tight_layout()
    
    return fig


def plot_average_metrics(metrics: Dict[str, Dict[str, float]], metrics_to_plot: List[str] = None):
    """
    Plot average metrics across different methods.
    
    Args:
        metrics (dict): Dictionary mapping method names to dictionaries of metrics
        metrics_to_plot (list): List of metric names to plot
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Determine which metrics to plot
    if metrics_to_plot is None:
        # Get all unique metrics
        metrics_to_plot = set()
        for method_metrics in metrics.values():
            metrics_to_plot.update(method_metrics.keys())
        metrics_to_plot = sorted(list(metrics_to_plot))
    
    # Create figure
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Extract values for this metric from each method
        methods = []
        values = []
        for method, method_metrics in metrics.items():
            if metric in method_metrics:
                methods.append(method)
                values.append(method_metrics[metric])
        
        # Plot bar chart
        colors = sns.color_palette("viridis", len(methods))
        ax.bar(methods, values, color=colors)
        
        # Add labels
        ax.set_title(metric)
        ax.set_ylabel("Value")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig


def plot_forgetting_curve(forgetting_values: Dict[str, List[float]], task_names: List[str] = None):
    """
    Plot forgetting curves for different methods.
    
    Args:
        forgetting_values (dict): Dictionary mapping method names to lists of forgetting values
        task_names (list): Names of the tasks
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Get number of tasks
    num_tasks = len(next(iter(forgetting_values.values())))
    
    # Create task names if not provided
    if task_names is None:
        task_names = [f"Task {i+1}" for i in range(num_tasks)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot forgetting curves for each method
    for method, values in forgetting_values.items():
        plt.plot(range(1, num_tasks), values, marker='o', linewidth=2, label=method)
    
    # Add labels and title
    plt.xlabel("Tasks Learned")
    plt.ylabel("Average Forgetting (%)")
    plt.title("Forgetting Curve for Different Methods")
    plt.xticks(range(1, num_tasks), [f"After Task {i+1}" for i in range(1, num_tasks)])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo visualization with random data
    num_tasks = 3
    performance = np.random.uniform(60, 100, size=(num_tasks, num_tasks))
    # Set upper triangle to NaN (tasks not seen yet)
    for i in range(num_tasks):
        for j in range(i+1, num_tasks):
            performance[i, j] = np.nan
    
    # Example forgetting matrix
    forgetting = np.zeros((num_tasks, num_tasks))
    for i in range(1, num_tasks):
        for j in range(i):
            forgetting[i, j] = performance[j, j] - performance[i, j]
    
    # Create demo plots
    task_names = ["Digits 0-4", "Digits 5-9", "Fashion MNIST"]
    
    perf_fig = plot_performance(performance, task_names)
    perf_fig.savefig("demo_performance.png")
    
    forget_fig = plot_forgetting(forgetting, task_names)
    forget_fig.savefig("demo_forgetting.png")
    
    logger.info("Created demo visualizations: demo_performance.png and demo_forgetting.png") 