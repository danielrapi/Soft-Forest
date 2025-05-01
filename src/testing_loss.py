############################ GOAL ###########################################
'''
    Run performance tests on loss to see if it is correct 
'''
#############################################################################



# Third-party imports
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from engine import train_model
from data_handling import load_processed_classification_public_data

from softensemble import SoftTreeEnsemble

# we will first look at a single tree

# first load in the data
data = load_processed_classification_public_data(name = "ann_thyroid", noise_level=0.0)
train_X_tensor = torch.tensor(data.x_train_processed, dtype=torch.float32)
train_y_tensor = torch.tensor(data.y_train_processed, dtype=torch.float32)
test_X_tensor = torch.tensor(data.x_test_processed, dtype=torch.float32)
test_y_tensor = torch.tensor(data.y_test_processed, dtype=torch.float32) 

# some printing statements
print("The shape of the X_train:", train_X_tensor.shape)
print("The shape of the X_test:", data.x_test_processed.shape)
print("The number of unique labels in train: ", np.unique(data.y_train_processed))
print("The number of unique labels in the test: ", np.unique(data.y_test_processed))


# there are 36 columns in the data set and there are [0, 1, 2] as the label

train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

input_dims = train_X_tensor.shape[1]
# this shit should be 36
print("The input dims are: ", input_dims)

print("The number of classes is:", data.num_classes)


# now we are ready to process in data loaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False)

# we can now create a model
    # Create model
model = SoftTreeEnsemble(
    num_trees=1,
    max_depth=2,
    leaf_dims=3,
    input_dim=36,
    combine_output=True,
    subset_selection=None,
    subset_share=None
)

# let's create a tensor that has the expected shape but with one operation
first_example = train_X_tensor[0]

# pass this through the model 
y = model(first_example)

print("The shape of y is: ", y.shape)

# at this point we have a 1,3 for each of the raw scores of the probailities 
# ok now we need to test the engine part 

# starting to train the model
test_loss, test_accuracy, test_auc = train_model(
    model=model,
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    epochs=1,
    learning_rate=0.05,
    device="cpu"
)