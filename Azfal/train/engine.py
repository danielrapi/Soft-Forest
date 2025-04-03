#################################################################################
'''
This is the engine for the soft_trees. This will be for the PyTorch Implementation.

'''

#################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import logging

################################################################################
# GOAL: Train the model
################################################################################

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001, device='cpu', optimizer=None):
    '''
        The training loop for the function. 

        Params:
            Model (SoftTree).
            Need to implement GPU functionality for later.
    
    '''
    # Define the optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  

    # Move model to device (GPU or CPU)
    model.to(device)
    # Training loop
    for epoch in range(epochs):
        
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            # Move data to device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # IMPORTANT FIX: Don't reshape the labels to 2D for CrossEntropyLoss
            # Instead, make sure they're 1D
            if len(labels.shape) > 1:
                labels = labels.squeeze()  # Convert [batch_size, 1] to [batch_size]
            
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            labels = labels.long()  # Convert to Long type for CrossEntropyLoss
            # Calculate the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            running_loss += loss.item()
            # Track accuracy or other metrics if needed
            # For classification, you can calculate accuracy
            predicted = outputs.argmax(dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_preds / total_preds

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        logging.info(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} | Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Fix the labels dimension
            if len(labels.shape) > 1:
                labels = labels.squeeze()  # Convert [batch_size, 1] to [batch_size]
                
            labels = labels.long()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy







