import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
import logging
import graphviz
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmoothStep(tf.keras.layers.Layer):
    """A smooth-step function.
    For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.
    """
    
    def __init__(self, gamma=1.0):
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def call(self, inputs):
        return tf.where(
            inputs <= self._lower_bound,
            tf.zeros_like(inputs),
            tf.where(
                inputs >= self._upper_bound,
                tf.ones_like(inputs),
                self._a3 * (inputs**3) + self._a1 * inputs + self._a0
            )
        )


class RandomForestEnsemble(tf.keras.layers.Layer):
    """An ensemble of soft decision trees with random feature selection at each node.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        - Fully vectorized implementation treating the ensemble as one "super" tree
        - Automatic detection of regression vs classification based on leaf_dims
        - Random feature selection at each node following Random Forest literature
        - For regression: uses p/3 features per node
        - For classification: uses sqrt(p) features per node
    
    Input:
        An input tensor of shape = (batch_size, num_features)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self,
                 num_trees,
                 max_depth,
                 leaf_dims,
                 features_per_node=None,  # Optional: override default feature selection count
                 activation='sigmoid',
                 node_index=0,
                 internal_eps=0,
                 kernel_regularizer=tf.keras.regularizers.L2(0.0),
                 combine_output=True):
        super(RandomForestEnsemble, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.internal_eps = internal_eps
        self.kernel_regularizer = kernel_regularizer
        self.combine_output = combine_output
        self.features_per_node = features_per_node
        
        # Log initialization parameters
        logger.info(f"Initializing RandomForestEnsemble node {node_index}:")
        logger.info(f"  - Number of trees: {num_trees}")
        logger.info(f"  - Max depth: {max_depth}")
        logger.info(f"  - Leaf dimensions: {leaf_dims}")
        logger.info(f"  - User-specified features per node: {features_per_node}")
        
        # Automatically determine if regression or classification based on leaf_dims
        # If leaf_dims is 1 or a tuple/list ending in 1, it's regression
        # Otherwise (e.g., leaf_dims > 1 for multi-class), it's classification
        # Relevant for feature selection 
        self.is_regression = (isinstance(leaf_dims, (int, float)) and leaf_dims == 1) or (
            isinstance(leaf_dims, (tuple, list)) and leaf_dims[-1] == 1)
        logger.info(f"  - Task type: {'regression' if self.is_regression else 'classification'}")
        
        if not self.leaf:
            self.dense_layer = tf.keras.layers.Dense(
                self.num_trees,
                kernel_regularizer=self.kernel_regularizer,
                activation='sigmoid',
            )
            self.left_child = RandomForestEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                features_per_node=features_per_node,
                activation=activation,
                node_index=2*self.node_index+1,
                kernel_regularizer=self.kernel_regularizer,
                combine_output=combine_output
            )
            self.right_child = RandomForestEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                features_per_node=features_per_node,
                activation=activation,
                node_index=2*self.node_index+2,
                kernel_regularizer=self.kernel_regularizer,
                combine_output=combine_output
            )

    def build(self, input_shape):
        if self.leaf:
            self.leaf_weight = self.add_weight(
                shape=[1, self.leaf_dims, self.num_trees],
                trainable=True,
                name="Node-"+str(self.node_index))
            logger.info(f"Building leaf node {self.node_index}")
            logger.info(f"  - Leaf weight shape: {self.leaf_weight.shape}")
        else:
            # Get the number of features as an integer
            num_features = int(input_shape[-1]) # input_shape = (batch_size, num_features)
            logger.info(f"\nBuilding internal node {self.node_index}")
            logger.info(f"  - Total available features: {num_features}")
            
            # Random Forest feature selection logic based on LITERATURE STANDARDS:
            # - For regression: use p/3 features (where p is total number of features)
            # - For classification: use sqrt(p) features
            # - User can override with features_per_node parameter
            if self.features_per_node is None:
                if self.is_regression:
                    self.features_per_node = max(1, num_features // 3)  # p/3 for regression
                    logger.info(f"  - Using p/3 features for regression: {self.features_per_node}")
                else:
                    self.features_per_node = max(1, int(np.sqrt(num_features)))  # sqrt(p) for classification
                    logger.info(f"  - Using sqrt(p) features for classification: {self.features_per_node}")
            
            # Ensure we don't select more features than available
            self.features_per_node = min(self.features_per_node, num_features)
            self.num_features = num_features
            
            # Log feature selection details
            logger.info(f"  - Number of features selected: {self.features_per_node}")
            
            # Create feature indices using numpy (not TensorFlow tensors)
            # This is critical for graph mode compatibility
            np.random.seed(self.node_index)  # Make selections deterministic per node - OPTIONAL
            
            self.feature_indices = np.sort(np.random.choice(
                num_features, self.features_per_node, replace=False))
            
            logger.info(f"  - Selected feature indices: {self.feature_indices.tolist()}")
            
            # Build dense layer 
            self.dense_layer.build(tf.TensorShape([None, self.features_per_node]))
            
            # Logging Details - OPTIONAL
            # Log other details in eager mode
            if tf.executing_eagerly():
                dense_weights = self.dense_layer.get_weights()
                if dense_weights:
                    logger.info(f"  - Dense layer weight shape: {dense_weights[0].shape}")
                    logger.info(f"  - Feature reduction ratio: {self.features_per_node/num_features:.2%}")
                    
                    # Log weight statistics
                    w = dense_weights[0]
                    logger.info(f"  - Weight stats - Mean: {w.mean():.4f}, Std: {w.std():.4f}")
                    logger.info(f"  - Weight range - Min: {w.min():.4f}, Max: {w.max():.4f}")

    def call(self, input, prob=1.0):
        if not self.leaf:
            # Create selection tensor during forward pass - compatible with graph mode
            selected_input = tf.gather(input, self.feature_indices, axis=-1)
            
            current_prob = tf.keras.backend.clip(
                self.dense_layer(selected_input), 
                self.internal_eps, 
                1-self.internal_eps
            )
            
            # Log probability statistics for root node, but only if in eager execution
            if self.node_index == 0 and tf.executing_eagerly():
                logger.debug(f"  - Input shape: {input.shape}")
                logger.debug(f"  - Selected features shape: {selected_input.shape}")
                logger.debug(f"  - Split probabilities - Mean: {tf.reduce_mean(current_prob):.4f}")
                logger.debug(f"  - Split probabilities range - Min: {tf.reduce_min(current_prob):.4f}, Max: {tf.reduce_max(current_prob):.4f}")
            
            return self.left_child(input, current_prob * prob) + self.right_child(input, (1 - current_prob) * prob)
        else:
            output = tf.expand_dims(prob, axis=1) * self.leaf_weight
            if self.combine_output:
                output = tf.math.reduce_sum(output, axis=2)
            return output

    def get_config(self):
        config = super(RandomForestEnsemble, self).get_config()
        config.update({
            "num_trees": self.num_trees,
            "max_depth": self.max_depth,
            "leaf_dims": self.leaf_dims,
            "features_per_node": self.features_per_node,
            "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            "combine_output": self.combine_output
        })
        return config

    #OPTIONAL FOR VISUALIZATION - TO BE TESTED

    # def visualize_tree(self, tree_index=0, filename="tree_visualization"):
    #     """Visualizes the tree structure using graphviz.
        
    #     Args:
    #         tree_index: Which tree in the ensemble to visualize (default: 0)
    #         filename: Name of the output file (without extension)
        
    #     Returns:
    #         Graphviz dot object that can be rendered
    #     """
    #     dot = graphviz.Digraph(comment='Soft Decision Tree')
    #     dot.attr(rankdir='TB')  # Top to bottom layout
        
    #     def add_nodes_edges(node, parent_id=None):
    #         current_id = str(node.node_index)
            
    #         if node.leaf:
    #             # For leaf nodes, show weight statistics
    #             weights = node.leaf_weight.numpy()
    #             mean_weight = np.mean(weights[..., tree_index])
    #             std_weight = np.std(weights[..., tree_index])
    #             label = f"Leaf {node.node_index}\nμ={mean_weight:.2f}\nσ={std_weight:.2f}"
    #             dot.node(current_id, label, shape='box')
    #         else:
    #             # For internal nodes, show selected features
    #             features = node.feature_indices
    #             weights = node.dense_layer.get_weights()[0]
    #             mean_weight = np.mean(weights[..., tree_index])
    #             label = f"Node {node.node_index}\nFeatures: {features}\nμ_w={mean_weight:.2f}"
    #             dot.node(current_id, label, shape='oval')
                
    #             # Recursively add children
    #             add_nodes_edges(node.left_child, current_id)
    #             add_nodes_edges(node.right_child, current_id)
            
    #         # Add edge from parent if not root
    #         if parent_id is not None:
    #             dot.edge(parent_id, current_id)
        
    #     # Start the recursive visualization from root
    #     add_nodes_edges(self)
        
    #     # Save the visualization
    #     dot.render(filename, view=True, format='png')
    #     return dot

    # def plot_feature_importance(self, feature_names=None):
    #     """Plots feature importance based on usage frequency in the tree.
        
    #     Args:
    #         feature_names: Optional list of feature names for better visualization
    #     """
    #     if feature_names is None:
    #         feature_names = [f"Feature_{i}" for i in range(self.input_shape[-1])]
        
    #     def collect_features(node):
    #         if node.leaf:
    #             return []
    #         features = node.feature_indices
    #         return features + collect_features(node.left_child) + collect_features(node.right_child)
        
    #     # Collect all used features
    #     all_features = collect_features(self)
        
    #     # Count feature usage
    #     feature_counts = np.bincount(all_features, minlength=len(feature_names))
        
    #     # Plot
    #     plt.figure(figsize=(12, 6))
    #     plt.bar(range(len(feature_counts)), feature_counts)
    #     plt.xticks(range(len(feature_counts)), feature_names, rotation=45, ha='right')
    #     plt.xlabel('Features')
    #     plt.ylabel('Usage Count')
    #     plt.title('Feature Usage in Tree')
    #     plt.tight_layout()
    #     plt.show()

    # def print_tree_structure(self, tree_index=0, prefix=""):
    #     """Prints a text representation of the tree structure.
        
    #     Args:
    #         tree_index: Which tree in the ensemble to visualize (default: 0)
    #         prefix: String prefix for pretty printing
    #     """
    #     if self.leaf:
    #         weights = self.leaf_weight.numpy()
    #         mean_weight = np.mean(weights[..., tree_index])
    #         print(f"{prefix}└── Leaf {self.node_index} (μ={mean_weight:.2f})")
    #     else:
    #         features = self.feature_indices
    #         weights = self.dense_layer.get_weights()[0]
    #         mean_weight = np.mean(weights[..., tree_index])
    #         print(f"{prefix}├── Node {self.node_index}")
    #         print(f"{prefix}│   Features: {features}")
    #         print(f"{prefix}│   Mean weight: {mean_weight:.2f}")
            
    #         # Print children
    #         self.left_child.print_tree_structure(tree_index, prefix + "│   ")
    #         self.right_child.print_tree_structure(tree_index, prefix + "    ")


class MultitaskRandomForestEnsemble(RandomForestEnsemble):
    """Multitask version of the RandomForestEnsemble.
    
    Inherits from RandomForestEnsemble and adds multitask-specific functionality.
    Each tree in the ensemble can handle multiple related tasks simultaneously.
    """
    
    def __init__(self,
                 num_trees,
                 max_depth,
                 num_tasks,
                 leaf_dims,
                 activation='sigmoid',
                 node_index=0,
                 depth_index=0,
                 internal_eps=0,
                 alpha=1.0,
                 power=2.0,
                 name='Node-Root',
                 **kwargs):
        super(MultitaskRandomForestEnsemble, self).__init__(
            num_trees=num_trees,
            max_depth=max_depth,
            leaf_dims=leaf_dims,
            activation=activation,
            node_index=node_index,
            internal_eps=internal_eps,
            name=name,
            **kwargs
        )
        self.num_tasks = num_tasks
        self.task_size = (int)(leaf_dims/num_tasks)
        self.depth_index = depth_index
        self.alpha = alpha
        self.power = power

    def get_config(self):
        config = super(MultitaskRandomForestEnsemble, self).get_config()
        config.update({
            "num_tasks": self.num_tasks,
            "depth_index": self.depth_index,
            "alpha": self.alpha,
            "power": self.power
        })
        return config 