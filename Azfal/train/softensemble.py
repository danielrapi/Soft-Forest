import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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

            #print(f"The mask looks like: {self.mask}")

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
    def plot_tree(self, ax=None, x=0, y=0, width=1.0, depth=0):
        """
        Plots the tree structure recursively.
        
        Parameters:
        - ax: matplotlib axis to plot on
        - x, y: coordinates of the current node
        - width: width of the current subtree
        - depth: current depth in the tree
        """
        if self.num_trees > 1:
           raise ValueError("This function is only supported for single tree models")

        # Create a new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title('Soft Decision Tree Structure')
            ax.axis('off')
            
        # Draw the current node
        node_radius = 0.03 * (self.max_depth - depth + 1)  # Larger nodes at top
        node_color = 'skyblue' if not self.leaf else 'lightgreen'
        circle = plt.Circle((x, self.max_depth - depth), node_radius, color=node_color, 
                           ec='black', zorder=10, alpha=0.8)
        ax.add_patch(circle)
        
        # Add node index
        ax.text(x, self.max_depth - depth, f"{self.node_index}", 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # If not a leaf, recursively plot children
        if not self.leaf:
            # Calculate positions for children
            left_x = x - width/2
            right_x = x + width/2
            child_y = y + 1
            
            # Draw edges to children
            ax.plot([x, left_x], [self.max_depth - depth, self.max_depth - (depth+1)], 
                   'k-', alpha=0.6, linewidth=1.5)
            ax.plot([x, right_x], [self.max_depth - depth, self.max_depth - (depth+1)], 
                   'k-', alpha=0.6, linewidth=1.5)
            
            # Add edge labels with background
            midpoint_left = ((x + left_x)/2, self.max_depth - depth - 0.5)
            midpoint_right = ((x + right_x)/2, self.max_depth - depth - 0.5)
            
            # Add small circles for the decision values
            decision_radius = 0.02
            left_circle = plt.Circle(midpoint_left, decision_radius, color='white', 
                                    ec='blue', zorder=9, alpha=0.9)
            right_circle = plt.Circle(midpoint_right, decision_radius, color='white', 
                                     ec='red', zorder=9, alpha=0.9)
            ax.add_patch(left_circle)
            ax.add_patch(right_circle)
            
            ax.text(midpoint_left[0], midpoint_left[1], "1", color='blue', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(midpoint_right[0], midpoint_right[1], "0", color='red', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add decision function info
            if hasattr(self, 'fc'):
                w = self.fc.weight.data.cpu().numpy().flatten()
                b = self.fc.bias.data.cpu().numpy().item()
                weight_str = ", ".join([f"{val:.2f}" for val in w])
                decision_text = f"w=[{weight_str}], b={b:.2f}"
                ax.text(x, self.max_depth - depth - 0.2, decision_text,
                       ha='center', va='top', fontsize=7, 
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
            # Recursively plot children
            self.right_child.plot_tree(ax, right_x, child_y, width/2, depth+1)
            self.left_child.plot_tree(ax, left_x, child_y, width/2, depth+1)
        else:
            # For leaf nodes, display the leaf weights
            weights = self.leaf_weights.detach().cpu().numpy().flatten()
            weight_str = ", ".join([f"{w:.2f}" for w in weights])
            ax.text(x, self.max_depth - depth - 0.15, f"w=[{weight_str}]", 
                   ha='center', va='top', fontsize=7,
                   bbox=dict(boxstyle="round,pad=0.3", fc="honeydew", ec="green", alpha=0.7))
        
        # Return the figure if this is the root call
        if depth == 0:
            plt.tight_layout()
            return ax.figure
