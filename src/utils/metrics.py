#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metrics module for the Continual Learning System.
This module provides functions for calculating and tracking 
performance metrics in continual learning.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def evaluate_performance(
    model,
    task_loaders: List[Any],
    device,
    use_task_id: bool = False
) -> np.ndarray:
    """
    Evaluate the performance of a model on all provided tasks.
    
    Args:
        model: PyTorch model to evaluate
        task_loaders (list): List of data loaders for each task
        device: Device to run the model on
        use_task_id (bool): Whether to provide task_id to the model's forward method
        
    Returns:
        np.ndarray: Vector of accuracies for each task
    """
    model.eval()
    accuracies = []
    
    for task_id, loader in enumerate(task_loaders):
        correct = 0
        total = 0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with or without task_id
            if use_task_id:
                outputs = model(inputs, task_id=task_id)
            else:
                outputs = model(inputs)
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate accuracy for this task
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        accuracies.append(accuracy)
    
    return np.array(accuracies)


def compute_forgetting(performance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the forgetting matrix from a performance matrix.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training steps
                                        (task i trained) and columns represent 
                                        performance on each task
    
    Returns:
        np.ndarray: Forgetting matrix where each element [i,j] represents how much
                   task j was forgotten after training on task i
    """
    num_tasks = performance_matrix.shape[0]
    forgetting_matrix = np.zeros_like(performance_matrix)
    
    for i in range(1, num_tasks):  # For each training step (except the first)
        for j in range(i):  # For each previously learned task
            # Forgetting is the difference between the best previous performance
            # and the current performance
            best_previous = np.max(performance_matrix[:i, j])
            forgetting_matrix[i, j] = max(0, best_previous - performance_matrix[i, j])
    
    return forgetting_matrix


def average_forgetting(forgetting_matrix: np.ndarray) -> List[float]:
    """
    Compute the average forgetting after each task.
    
    Args:
        forgetting_matrix (np.ndarray): Matrix where each element [i,j] represents
                                       how much task j was forgotten after training
                                       on task i
    
    Returns:
        list: Average forgetting after each task
    """
    num_tasks = forgetting_matrix.shape[0]
    avg_forgetting = []
    
    for i in range(1, num_tasks):  # For each training step (except the first)
        # Average forgetting for all previous tasks
        if i > 0:
            avg_forgetting.append(np.mean(forgetting_matrix[i, :i]))
    
    return avg_forgetting


def backward_transfer(performance_matrix: np.ndarray) -> List[float]:
    """
    Compute the backward transfer after each task.
    Backward transfer measures how training on later tasks affects 
    performance on earlier tasks.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training steps
                                        (task i trained) and columns represent 
                                        performance on each task
    
    Returns:
        list: Backward transfer after each task
    """
    num_tasks = performance_matrix.shape[0]
    bt_values = []
    
    for i in range(1, num_tasks):  # For each training step (except the first)
        # Sum of differences between current and original performance
        bt_sum = 0
        for j in range(i):  # For each previously learned task
            bt_sum += performance_matrix[i, j] - performance_matrix[j, j]
        
        # Average backward transfer
        bt_values.append(bt_sum / i if i > 0 else 0)
    
    return bt_values


def forward_transfer(performance_matrix: np.ndarray, random_performance: List[float]) -> List[float]:
    """
    Compute the forward transfer after each task.
    Forward transfer measures how training on earlier tasks affects 
    the initial performance on later tasks.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training steps
                                        (task i trained) and columns represent 
                                        performance on each task
        random_performance (list): Performance on each task with random initialization
        
    Returns:
        list: Forward transfer after each task
    """
    num_tasks = performance_matrix.shape[0]
    ft_values = []
    
    for i in range(1, num_tasks):  # For each task (except the first)
        # Performance on task i before training on it
        initial_perf = performance_matrix[i-1, i]
        # Performance on task i with random initialization
        random_perf = random_performance[i]
        
        # Forward transfer is the difference
        ft_values.append(initial_perf - random_perf)
    
    return ft_values


def learning_curve_area(learning_curves: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute the area under the learning curve for different methods.
    Higher area means faster learning.
    
    Args:
        learning_curves (dict): Dictionary mapping method names to lists of 
                               performance values during training
                               
    Returns:
        dict: Dictionary mapping method names to ALC values
    """
    alc_values = {}
    
    for method, curve in learning_curves.items():
        # Normalize curve to [0, 1]
        min_val = min(curve)
        max_val = max(curve)
        
        if max_val > min_val:
            normalized_curve = [(v - min_val) / (max_val - min_val) for v in curve]
        else:
            normalized_curve = [0.5] * len(curve)
        
        # Compute area under the curve
        alc = np.trapz(normalized_curve, dx=1.0/len(normalized_curve))
        alc_values[method] = alc
    
    return alc_values


def average_accuracy(performance_matrix: np.ndarray) -> float:
    """
    Compute the average accuracy across all tasks after training is complete.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training steps
                                        (task i trained) and columns represent 
                                        performance on each task
                                        
    Returns:
        float: Average final accuracy across all tasks
    """
    # Final performance is the last row of the matrix
    final_performance = performance_matrix[-1, :]
    return np.mean(final_performance)


def compute_metrics_summary(performance_matrix: np.ndarray, random_performance: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute a summary of continual learning metrics.
    
    Args:
        performance_matrix (np.ndarray): Matrix where rows represent training steps
                                        (task i trained) and columns represent 
                                        performance on each task
        random_performance (list, optional): Performance on each task with random initialization
        
    Returns:
        dict: Dictionary of metric values
    """
    # Compute forgetting
    forgetting_matrix = compute_forgetting(performance_matrix)
    avg_forget = np.mean(forgetting_matrix[-1, :-1]) if performance_matrix.shape[0] > 1 else 0.0
    
    # Compute backward transfer
    bt_values = backward_transfer(performance_matrix)
    avg_bt = np.mean(bt_values) if bt_values else 0.0
    
    # Compute forward transfer if random performance is provided
    avg_ft = 0.0
    if random_performance is not None:
        ft_values = forward_transfer(performance_matrix, random_performance)
        avg_ft = np.mean(ft_values) if ft_values else 0.0
    
    # Compute average accuracy
    avg_acc = average_accuracy(performance_matrix)
    
    # Return metrics summary
    return {
        'average_accuracy': avg_acc,
        'average_forgetting': avg_forget,
        'backward_transfer': avg_bt,
        'forward_transfer': avg_ft
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example performance matrix
    num_tasks = 3
    performance = np.array([
        [90.0, np.nan, np.nan],  # After task 1, only task 1 evaluated
        [85.0, 92.0, np.nan],    # After task 2, tasks 1-2 evaluated
        [82.0, 88.0, 94.0]       # After task 3, all tasks evaluated
    ])
    
    # Example random performance
    random_perf = [40.0, 42.0, 38.0]
    
    # Compute and print metrics
    forgetting_mat = compute_forgetting(performance)
    print("Forgetting matrix:")
    print(forgetting_mat)
    
    avg_forgetting_values = average_forgetting(forgetting_mat)
    print(f"Average forgetting after each task: {avg_forgetting_values}")
    
    bt_values = backward_transfer(performance)
    print(f"Backward transfer after each task: {bt_values}")
    
    ft_values = forward_transfer(performance, random_perf)
    print(f"Forward transfer after each task: {ft_values}")
    
    metrics_summary = compute_metrics_summary(performance, random_perf)
    print("Metrics summary:")
    for metric, value in metrics_summary.items():
        print(f"  {metric}: {value:.2f}") 