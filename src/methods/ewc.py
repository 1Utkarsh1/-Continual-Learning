#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Elastic Weight Consolidation (EWC) learner for continual learning.
EWC prevents catastrophic forgetting by penalizing changes to parameters 
that are important for previously learned tasks.

Reference:
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017).
"Overcoming catastrophic forgetting in neural networks."
Proceedings of the National Academy of Sciences, 114(13), 3521-3526.
"""

import os
import logging
import copy
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline import BaselineLearner

logger = logging.getLogger(__name__)


class EWCLearner(BaselineLearner):
    """
    Elastic Weight Consolidation (EWC) learner.
    Extends the baseline learner with a regularization term to prevent forgetting.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 0.001,
                lambda_ewc: float = 5000, fisher_sample_size: int = 200):
        """
        Initialize the EWC learner.
        
        Args:
            model (nn.Module): The neural network model
            device (torch.device): Device to run the model on
            learning_rate (float): Learning rate for the optimizer
            lambda_ewc (float): Regularization strength for EWC
            fisher_sample_size (int): Number of samples to estimate Fisher information
        """
        super().__init__(model, device, learning_rate)
        
        self.lambda_ewc = lambda_ewc
        self.fisher_sample_size = fisher_sample_size
        
        # Store parameters and Fisher information for each task
        self.fisher_matrices = {}  # Fisher information for each task
        self.optimal_parameters = {}  # Optimal parameters for each task
    
    def _compute_fisher_information(self, data_loader: DataLoader, task_id: int):
        """
        Compute the Fisher information matrix for the current task.
        The Fisher information measures how much the model parameters affect the output.
        
        Args:
            data_loader (DataLoader): Data loader for the current task
            task_id (int): ID of the current task
        """
        logger.info(f"Computing Fisher information matrix for task {task_id+1}")
        
        # Initialize Fisher information matrix
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Sample a subset of data for Fisher computation
        sample_count = 0
        
        for inputs, targets in data_loader:
            if sample_count >= self.fisher_sample_size:
                break
                
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.shape[0]
            
            # Get model outputs
            log_probs = F.log_softmax(self.model(inputs), dim=1)
            
            # Compute gradients for each sample in the batch
            for i in range(batch_size):
                if sample_count >= self.fisher_sample_size:
                    break
                
                sample_log_prob = log_probs[i, targets[i]]
                
                # Compute gradients
                self.optimizer.zero_grad()
                sample_log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data.pow(2) / self.fisher_sample_size
                
                sample_count += 1
        
        # Store the computed Fisher information
        self.fisher_matrices[task_id] = fisher
        
        logger.info(f"Fisher information matrix computed using {sample_count} samples")
    
    def _store_optimal_parameters(self, task_id: int):
        """
        Store the optimal parameters for the current task.
        
        Args:
            task_id (int): ID of the current task
        """
        logger.info(f"Storing optimal parameters for task {task_id+1}")
        
        optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()
        
        self.optimal_parameters[task_id] = optimal_params
    
    def _compute_ewc_loss(self):
        """
        Compute the EWC regularization loss based on stored Fisher information.
        This penalties changes to parameters that were important for previous tasks.
        
        Returns:
            torch.Tensor: EWC regularization loss
        """
        ewc_loss = 0
        
        for task_id in range(self.current_task):
            if task_id not in self.fisher_matrices or task_id not in self.optimal_parameters:
                continue
                
            for name, param in self.model.named_parameters():
                if name in self.fisher_matrices[task_id] and name in self.optimal_parameters[task_id]:
                    # Compute the squared difference between current and optimal parameters
                    # weighted by the Fisher information
                    fisher = self.fisher_matrices[task_id][name]
                    optimal_param = self.optimal_parameters[task_id][name]
                    ewc_loss += torch.sum(fisher * (param - optimal_param).pow(2)) / 2
        
        return ewc_loss * self.lambda_ewc
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             task_id: int, epochs: int, eval_freq: int = 1):
        """
        Train the model on a new task with EWC regularization.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            task_id (int): ID of the current task
            epochs (int): Number of training epochs
            eval_freq (int): Frequency of evaluation during training (in epochs)
        """
        # Update task info
        self.current_task = task_id
        self.seen_tasks = max(self.seen_tasks, task_id + 1)
        
        # Create a new optimizer for the new task with the original learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_task_loss = 0.0
            train_ewc_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Use tqdm for a nice progress bar
            train_pbar = tqdm(train_loader, desc=f"Task {task_id+1} - Epoch {epoch+1}/{epochs} [Train]")
            for inputs, targets in train_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                task_loss = self.criterion(outputs, targets)
                
                # Compute EWC regularization loss (for tasks > 0)
                ewc_loss = self._compute_ewc_loss() if task_id > 0 else 0
                
                # Total loss
                loss = task_loss + ewc_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                train_task_loss += task_loss.item() * inputs.size(0)
                if task_id > 0:
                    train_ewc_loss += ewc_loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': loss.item(), 
                    'task_loss': task_loss.item(),
                    'ewc_loss': ewc_loss.item() if task_id > 0 else 0,
                    'acc': 100.0 * train_correct / train_total
                })
            
            train_loss = train_loss / train_total
            train_task_loss = train_task_loss / train_total
            train_ewc_loss = train_ewc_loss / train_total if task_id > 0 else 0
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase (run every eval_freq epochs)
            if epoch % eval_freq == 0:
                val_loss, val_acc = self._evaluate_training(val_loader)
                
                logger.info(f"Task {task_id+1} - Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f} (Task: {train_task_loss:.4f}, EWC: {train_ewc_loss:.4f}), "
                          f"Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
        
        # Load the best model from this task's training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model for task {task_id+1} with validation loss: {best_val_loss:.4f}")
        
        # After training on this task, compute and store Fisher information and optimal parameters
        self._compute_fisher_information(train_loader, task_id)
        self._store_optimal_parameters(task_id)
    
    def save(self, path: str):
        """
        Save the model and EWC-specific information to a file.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert fisher matrices and optimal parameters to CPU tensors for serialization
        fisher_cpu = {}
        for task_id, task_fisher in self.fisher_matrices.items():
            fisher_cpu[task_id] = {name: tensor.cpu() for name, tensor in task_fisher.items()}
        
        params_cpu = {}
        for task_id, task_params in self.optimal_parameters.items():
            params_cpu[task_id] = {name: tensor.cpu() for name, tensor in task_params.items()}
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'seen_tasks': self.seen_tasks,
            'current_task': self.current_task,
            'fisher_matrices': fisher_cpu,
            'optimal_parameters': params_cpu,
            'lambda_ewc': self.lambda_ewc,
            'fisher_sample_size': self.fisher_sample_size
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the model and EWC-specific information from a file.
        
        Args:
            path (str): Path to load the model from
        """
        if not os.path.exists(path):
            logger.error(f"Model file not found: {path}")
            return
        
        # Load model state
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.seen_tasks = checkpoint['seen_tasks']
        self.current_task = checkpoint['current_task']
        
        # Load EWC-specific information
        self.lambda_ewc = checkpoint.get('lambda_ewc', self.lambda_ewc)
        self.fisher_sample_size = checkpoint.get('fisher_sample_size', self.fisher_sample_size)
        
        # Load Fisher matrices and optimal parameters, moving them to the correct device
        if 'fisher_matrices' in checkpoint:
            self.fisher_matrices = {}
            for task_id, task_fisher in checkpoint['fisher_matrices'].items():
                self.fisher_matrices[int(task_id)] = {name: tensor.to(self.device) 
                                                    for name, tensor in task_fisher.items()}
        
        if 'optimal_parameters' in checkpoint:
            self.optimal_parameters = {}
            for task_id, task_params in checkpoint['optimal_parameters'].items():
                self.optimal_parameters[int(task_id)] = {name: tensor.to(self.device) 
                                                       for name, tensor in task_params.items()}
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("This is an implementation of Elastic Weight Consolidation for continual learning.")
    logger.info("Reference: Kirkpatrick et al. (2017) - 'Overcoming catastrophic forgetting in neural networks'") 