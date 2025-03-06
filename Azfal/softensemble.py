import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SoftTreeEnsemble(torch.nn.Module):

    def __init__(self, num_trees, max_depth, leaf_dims, input_dim, activation='sigmoid', node_index=0,
                 internal_eps = 0, combine_output = True, subset_selection = False):
        """
          # The max depth will determine the number of nodes that we have
          # This will be 2^{d+1} - 1 where d is the edges from root to leaf
          # This will be the dimensionality of the vector returned by the tree
          # should match the number of classes in the problem but can be more
          # if more then pass through final activation to resize
          # can only use this if we output raw score
          # s1.right_child.right_child.node_index
              This will grab the node from root, to the right_child, and to that right child


          # if combine_output is true we get (batch_size, leaf_dim)
          # if false then we get (batch_size, num_trees, leaf_dim)

          # subset selection:
              This will be a boolean value that if true we perform the randomization of feature selection in a random forrest


        """
        super(SoftTreeEnsemble, self).__init__()
        self.num_trees = num_trees
        self.combine_output = combine_output
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.activation = activation
        self.node_index = node_index
        self.internal_eps = internal_eps
        self.leaf = node_index >= 2**max_depth-1
        self.subset_selection = subset_selection
        # self.device = device
        

        # instatiate through recursion
        # we will build the tree structure first
        if not self.leaf:

          # for each node we need to make a FC layer w'x
          # takes as input the input dimension
          # and outputs the probabilities for each class
          # output dim is one to resemble a tabular data structure

          # to extend this to multiple trees we can do the following 
          # we have each level of the nodes but now extended to the number of 
          # trees 


          self.fc = nn.Linear(input_dim, num_trees)

          # we need to choose a feature of values
          if self.subset_selection:
            # create a matrix of the same row as the rows in W
            # for now will not be part of the gradient computation
            # we will keep sqrt(p) for now 
            num_features = math.floor(math.sqrt(input_dim))


            temp = torch.ones(input_dim, requires_grad=False)

            zero_indices = torch.randperm(temp.shape[0])[:num_features]
            temp[zero_indices] = 0

            self.mask = nn.Parameter(temp, requires_grad=False)

            print(f"The mask looks like: {self.mask}")

            # now we need to elemntwise multiply




          # builds out the left and the right child for a balanced tree
          # also gives is a node index
          self.left_child = SoftTreeEnsemble(
              num_trees, max_depth, leaf_dims, 
              input_dim, activation, 2*node_index+1 ,
              combine_output = self.combine_output, subset_selection = self.subset_selection
              )
          


          self.right_child = SoftTreeEnsemble(
              num_trees, max_depth, leaf_dims, input_dim ,activation, 
              2*node_index+2,combine_output = self.combine_output,
              subset_selection = self.subset_selection
              )
          


        else:
          # creates weights for the leaf nodes for voting
          # we also need to add this for the multiple trees
          self.leaf_weights = nn.Parameter(
              torch.randn(1, self.leaf_dims, self.num_trees, requires_grad=True) * 0.1
              )
         



    # then we call the actual forward pass of the tree
    def forward(self, x, prob=1.0):
        """
            This runs the forward class of the model
        """
        # print(self.combine_output)

        # we first check if it is a leaf or not
        if not self.leaf:
          # apply the hadamard
          if self.subset_selection:
            masked_weights = self.fc.weight * self.mask

            # print(f"THE MASKED WEIGHTS ARE {masked_weights}")
            current_prob = torch.clamp(torch.sigmoid(F.linear(x, masked_weights, self.fc.bias)), min=self.internal_eps, max=1-self.internal_eps)
            
          else:
            # we make sure that the decision is not hard 1 or 0
            current_prob = torch.clamp(torch.sigmoid(self.fc(x)), min=self.internal_eps, max=1-self.internal_eps)
            # return the probability for going to the left or the right
          return self.left_child(x, prob*current_prob) + self.right_child(x, prob*(1-current_prob))

          # what do we do when we get to a leaf
        else:


          # element wise product between the probability that the data point gets to the leaf
          # to the weight that the leaf has
          # enable prob to broad cast
          output = prob.unsqueeze(1) * self.leaf_weights
          

          if self.combine_output:
            output = torch.sum(output, dim=2)
            # rint(output) 

            
          # assert len(output.shape) == 3 
          return output
