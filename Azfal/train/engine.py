#################################################################################
'''
This is the engine for the soft_trees. This will be for the PyTorch Implementation.

'''

#################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
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
    # just extending to GPU capabilities
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # log this in the file 
    logging.info(f"Running on a: {device}")

    # Define the optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  
    
    # Move model to device (GPU or CPU)
    model.to(device)
    #Training loop
    for epoch in range(epochs):
        if (epoch + 1) % 5 == 0:
            print(f"Running EPOCH {epoch + 1}")
        
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
            
            # For classification, you can calculate accuracy

            predicted = outputs.argmax(dim=1)
            
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_preds / total_preds

        test_loss, test_accuracy, test_auc = evaluate(model, test_loader, criterion, device)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} | Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
    return test_loss, test_accuracy, test_auc


def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0.0
    total_preds = 0.0
    
    # For AUC calculation
    all_labels = []
    all_probs = []

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
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            
            # Store probabilities and labels for AUC
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = running_loss / len(test_loader)
    accuracy = correct_preds / total_preds
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Handle multi-class AUC calculation
    if all_probs.shape[1] > 2:
        # Use macro averaging for multi-class
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    else:
        # For binary classification, use the probability of class 1
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        
    return avg_loss, accuracy, auc
