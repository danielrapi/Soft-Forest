#################################################################################
'''
This is the engine for the soft_trees. This will be for the PyTorch Implementation.

'''

#################################################################################
from sklearn.preprocessing import label_binarize
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
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            # Move data to device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure labels are 1D and of type long
            if len(labels.shape) > 1:
                labels = labels.squeeze()  # Convert [batch_size, 1] to [batch_size]
            labels = labels.long()  # Convert to Long type for CrossEntropyLoss
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
            
        avg_loss = running_loss / len(train_loader)
        accuracy = correct_preds / total_preds

        # Evaluate on test set
        model.eval()  # Ensure model is in eval mode
        test_loss, test_accuracy, test_auc = evaluate(model, test_loader, criterion, device)
        
        # Log metrics
        logging.info(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} | Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
        
        # Verify no data leakage
        if test_accuracy > 0.9 and accuracy < 0.5:
            logging.warning(f"High test accuracy ({test_accuracy:.4f}) with low training accuracy ({accuracy:.4f}) - potential data leakage or model issue")
    
    return test_loss, test_accuracy, test_auc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    # Ensure model is in eval mode
    model.eval()
    
    running_loss = 0.0
    correct_preds = 0.0
    total_preds = 0.0
    
    # For AUC calculation
    all_labels = []
    all_probs = []
    
    # Track predictions for debugging
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Ensure labels are 1D and of type long
            if len(labels.shape) > 1:
                labels = labels.squeeze()
            labels = labels.long()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store predictions and labels for debugging
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
            # Update metrics
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
        classes = np.arange(all_probs.shape[1])
        all_labels_bin = label_binarize(all_labels, classes=classes)
        auc = roc_auc_score(all_labels_bin, all_probs, multi_class='ovr', average='macro')
    else:
        # For binary classification, use the probability of class 1
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    
    # Debug information
    logging.debug(f"Test set predictions distribution: {dict(zip(*np.unique(all_predictions, return_counts=True)))}")
    logging.debug(f"Test set true labels distribution: {dict(zip(*np.unique(all_true_labels, return_counts=True)))}")
    
    # Verify predictions make sense
    if accuracy > 0.9 and len(np.unique(all_predictions)) == 1:
        logging.warning("Model is predicting the same class for all samples with high accuracy - potential issue")
    
    return avg_loss, accuracy, auc
