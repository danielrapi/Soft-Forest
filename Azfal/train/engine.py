#################################################################################
'''
This is the engine for the soft_trees. This will be for the PyTorch Implementation.

'''

#################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
################################################################################
# GOAL: Train the model
################################################################################

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.001, device='cpu'):

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
    


    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  

    # Move model to device (GPU or CPU)
    model.to(device)

    # Training loop
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

                    # Ensure labels are of the correct shape (2D tensor)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)  # Change shape from [batch_size] to [batch_size, 1]

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # trying to squeeze to make the compatible shape in pytorch
            outputs = outputs.squeeze(-1)
            # print("THE SHAPE OF THE OUTPUTS IS")
            # print(outputs.shape)
            # print(outputs)
            labels = labels.long()
            # we need to also fix the labels 
            labels = labels.squeeze(-1)            
            # print("THE SHAPE OF THE labels IS")
            # print(labels.shape)
            # print(labels)
            # Calculate the loss
            loss = criterion(outputs, labels)


            # exit()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            running_loss += loss.item()



            # Track accuracy or other metrics if needed
            # For classification, you can calculate accuracy

            predicted = outputs.argmax(dim=1)
            correct_preds += (predicted == labels.squeeze()).sum().item()
            total_preds += labels.size(0)


        avg_loss = running_loss / len(train_loader)
        accuracy = correct_preds / total_preds
        


        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        logging.info(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f} | Training Accuracy: {accuracy} | Test Lost: {test_loss:.4f} | Test Accuracy {test_accuracy:.4f}")




def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_preds = 0.0
    total_preds = 0.0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1) 

            # print("THE SHAPES OF LABELS")
            labels = labels.long()
            labels = labels.squeeze(-1)
            # print(labels.shape)

            # Forward pass

            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            # print("THE SHAPES OF THE OUTPUTS")
            # print(outputs.shape)

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # calculate the accuracy as well
            predicted = outputs.argmax(dim=1)
            correct_preds += (predicted == labels.squeeze()).sum().item()
            total_preds += labels.size(0)



    return (running_loss / len(test_loader)), (correct_preds / total_preds)








