#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline learner that simply fine-tunes the model on new tasks.
This implementation will demonstrate catastrophic forgetting.
"""

import os
import logging
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaselineLearner:
    """
    Naive fine-tuning implementation that trains sequentially on tasks.
    Used as a baseline to demonstrate catastrophic forgetting.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, learning_rate: float = 0.001):
        """
        Initialize the baseline learner.
        
        Args:
            model (nn.Module): The neural network model
            device (torch.device): Device to run the model on
            learning_rate (float): Learning rate for the optimizer
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Keep track of tasks
        self.current_task = 0
        self.seen_tasks = 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             task_id: int, epochs: int, eval_freq: int = 1):
        """
        Train the model on a new task.
        
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
                loss = self.criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({'loss': loss.item(), 
                                      'acc': 100.0 * train_correct / train_total})
            
            train_loss = train_loss / train_total
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase (run every eval_freq epochs)
            if epoch % eval_freq == 0:
                val_loss, val_acc = self._evaluate_training(val_loader)
                
                logger.info(f"Task {task_id+1} - Epoch {epoch+1}/{epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save the best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
        
        # Load the best model from this task's training
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Loaded best model for task {task_id+1} with validation loss: {best_val_loss:.4f}")
    
    def _evaluate_training(self, val_loader: DataLoader) -> tuple:
        """
        Evaluate the model on validation data during training.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (validation loss, validation accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        
        return val_loss, val_acc
    
    def evaluate(self, test_loader: DataLoader, task_id: Optional[int] = None) -> float:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            task_id (int, optional): ID of the task to evaluate
            
        Returns:
            float: Test accuracy as a percentage
        """
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()
        
        test_acc = 100.0 * test_correct / test_total
        
        return test_acc
    
    def save(self, path: str):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'seen_tasks': self.seen_tasks,
            'current_task': self.current_task
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the model from a file.
        
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
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logger.info("This is a baseline learner that demonstrates catastrophic forgetting.")
    logger.info("It should be used as a comparison point for more advanced continual learning methods.") 