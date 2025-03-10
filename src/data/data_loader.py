#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading module for the Continual Learning System.
Handles loading and preprocessing of different datasets and task sequences.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Union, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

logger = logging.getLogger(__name__)


def get_dataset(dataset_name: str, root: str = './data', train: bool = True, transform=None):
    """
    Get the specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('mnist', 'fashion_mnist', 'kmnist', etc.)
        root (str): Root directory for dataset storage
        train (bool): Whether to load the training set or the test set
        transform: Transformations to apply to the data
        
    Returns:
        torch.utils.data.Dataset: The requested dataset
    """
    dataset_name = dataset_name.lower()
    
    # Create directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    # Default transformations if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST-like normalization
        ])
    
    # Load dataset based on name
    if dataset_name == 'mnist':
        dataset = torchvision.datasets.MNIST(
            root=root, train=train, download=True, transform=transform
        )
    elif dataset_name == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(
            root=root, train=train, download=True, transform=transform
        )
    elif dataset_name == 'kmnist':
        dataset = torchvision.datasets.KMNIST(
            root=root, train=train, download=True, transform=transform
        )
    elif dataset_name == 'cifar10':
        if transform is None:
            # Default CIFAR10 transformation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform
        )
    elif dataset_name == 'cifar100':
        if transform is None:
            # Default CIFAR100 transformation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        
        dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def create_class_subset(dataset, classes: Union[List[int], str]):
    """
    Create a subset of a dataset containing only the specified classes.
    
    Args:
        dataset: PyTorch dataset
        classes: List of class indices to include, or 'all' to include all classes
        
    Returns:
        tuple: (Subset of the dataset, list of classes included)
    """
    if classes == 'all':
        # If we want all classes, return the whole dataset and a list of all class indices
        if hasattr(dataset, 'classes'):
            all_classes = list(range(len(dataset.classes)))
        else:
            # Try to infer the number of classes
            targets = dataset.targets if hasattr(dataset, 'targets') else dataset.targets
            all_classes = list(set(targets.numpy() if torch.is_tensor(targets) else targets))
        
        return dataset, all_classes
    
    # Get targets from the dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'target'):
        targets = dataset.target
    else:
        # For datasets that store targets differently (e.g., as part of __getitem__)
        # We'll need to iterate through the dataset
        targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    
    # Convert targets to numpy array for easier filtering
    if torch.is_tensor(targets):
        targets = targets.numpy()
    elif not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    # Get indices for the requested classes
    indices = [i for i, label in enumerate(targets) if label in classes]
    
    # Create and return the subset
    return Subset(dataset, indices), classes


def split_dataset(dataset, val_split: float = 0.1):
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: PyTorch dataset
        val_split (float): Fraction of data to use for validation
        
    Returns:
        tuple: (training subset, validation subset)
    """
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


def get_task_sequence(task_configs: List[Dict[str, Any]], batch_size: int = 32):
    """
    Create data loaders for a sequence of tasks.
    
    Args:
        task_configs (list): List of task configuration dictionaries
        batch_size (int): Batch size for data loaders
        
    Returns:
        list: List of task data dictionaries, each containing name, classes, and data loaders
    """
    # Data root directory is in the project root
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    
    task_data = []
    
    for task_config in task_configs:
        dataset_name = task_config['dataset']
        task_name = task_config['name']
        classes = task_config['classes']
        
        logger.info(f"Loading task {task_name} (dataset: {dataset_name}, classes: {classes})")
        
        # Get training and test datasets
        train_dataset = get_dataset(dataset_name, root=data_root, train=True)
        test_dataset = get_dataset(dataset_name, root=data_root, train=False)
        
        # Create class-specific subsets if needed
        train_subset, actual_classes = create_class_subset(train_dataset, classes)
        test_subset, _ = create_class_subset(test_dataset, classes)
        
        # Split training set into train and validation
        train_split, val_split = split_dataset(train_subset)
        
        # Create data loaders
        train_loader = DataLoader(
            train_split, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=2, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_split, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_subset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2, 
            pin_memory=True
        )
        
        # Store task data
        task_data.append({
            'name': task_name,
            'dataset': dataset_name,
            'classes': actual_classes,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        })
        
        logger.info(f"Loaded task {task_name} with {len(train_split)} training, "
                   f"{len(val_split)} validation, and {len(test_subset)} test samples")
    
    return task_data


if __name__ == '__main__':
    # Test the data loading functionality
    logging.basicConfig(level=logging.INFO)
    
    # Example task sequence
    task_configs = [
        {'name': 'mnist_0_4', 'dataset': 'mnist', 'classes': [0, 1, 2, 3, 4]},
        {'name': 'mnist_5_9', 'dataset': 'mnist', 'classes': [5, 6, 7, 8, 9]}
    ]
    
    task_data = get_task_sequence(task_configs)
    
    # Print information about the tasks
    for i, task in enumerate(task_data):
        logger.info(f"Task {i+1}: {task['name']}")
        logger.info(f"  Classes: {task['classes']}")
        logger.info(f"  Training samples: {len(task['train_loader'].dataset)}")
        logger.info(f"  Validation samples: {len(task['val_loader'].dataset)}")
        logger.info(f"  Test samples: {len(task['test_loader'].dataset)}")
        
        # Get a batch of data
        images, labels = next(iter(task['train_loader']))
        logger.info(f"  Batch shape: {images.shape}, Labels: {labels.numpy()[:5]} ...") 