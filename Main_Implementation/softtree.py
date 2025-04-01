import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class SoftTree(torch.nn.Module):

    def __init__(self, max_depth, leaf_dims, input_dim, activation='sigmoid', node_index=0,
                 internal_eps = 0):
        """
          Implmements a soft tree for a single tree
          max_depth: the maximum depth of the tree
          leaf_dims: the number of dimensions of the leaf nodes
          input_dim: the dimension of the input data
          activation: the activation function to use
          node_index: the index of the node
          internal_eps: the epsilon value to use for the internal nodes

        """
        super(SoftTree, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.activation = activation
        self.node_index = node_index
        self.internal_eps = internal_eps
        self.leaf = node_index >= 2**max_depth-1
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

            self.fc = nn.Linear(input_dim, 1)

            # builds out the left and the right child for a balanced tree
            # also gives is a node index
            self.left_child = SoftTree(
                max_depth, leaf_dims, 
                input_dim, activation, 2*node_index+1 ,
                )

            self.right_child = SoftTree(
                max_depth, leaf_dims, input_dim ,activation, 
                2*node_index+2,
                )

        else:
          # creates weights for the leaf nodes for voting
          # we also need to add this for the multiple trees
          self.leaf_weights = nn.Parameter(
              torch.randn(1, self.leaf_dims, requires_grad=True) * 0.1
              )

    # then we call the actual forward pass of the tree
    def forward(self, x, prob=1.0):
        """
            This runs the forward class of the model
        """
        # we first check if it is a leaf or not
        if not self.leaf:

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
          
          # print(output) 
 
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

# test the soft tree
if __name__ == "__main__":
  tree = SoftTree(max_depth=3, leaf_dims=1, input_dim=2)
  #x = torch.randn(1, 2)
  tree.plot_tree().savefig('tree.png')

