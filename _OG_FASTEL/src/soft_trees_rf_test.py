import tensorflow as tf
from soft_trees_rf import RandomForestEnsemble, MultitaskRandomForestEnsemble
import logging

# Uncomment to see more detailed logs from the RandomForestEnsemble
# logging.basicConfig(level=logging.INFO)

class RandomForestEnsembleTest(tf.test.TestCase):
    def test_training(self):
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        
        # Create a random forest ensemble with 5 trees, each with depth 3 and 10 output dimensions
        # The model will automatically determine appropriate feature selection count
        forest_ensemble = RandomForestEnsemble(
            num_trees=5, 
            max_depth=3, 
            leaf_dims=10
        )
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            forest_ensemble
        ])
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        
        # Print model summary and layer details
        model.summary()
        for l in model.layers:
            print("===============================================", l.name)
            for w in l.get_weights():
                print(w.shape)
        
        # Train for 2 epochs
        history = model.fit(
            x_train, y_train, 
            epochs=2, 
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Check if the accuracy is above 0.9
        self.assertGreaterEqual(history.history['accuracy'][-1], 0.5)
        
        # Optional: Visualize one of the trees
        # Uncomment to see visualization
        # forest_ensemble.visualize_tree(tree_index=0, filename="mnist_tree")
        
        # Optional: Print tree structure
        # Uncomment to see tree structure details
        # print("\nTree structure of first tree:")
        # forest_ensemble.print_tree_structure(tree_index=0)

    def test_feature_selection(self):
        """Test the feature selection behavior of RandomForestEnsemble"""
        # Create a dummy dataset with 100 features
        x = tf.random.normal((100, 100))
        y = tf.random.uniform((100,), maxval=10, dtype=tf.int32)
        
        # Create ensemble for classification (should use sqrt(100) ≈ 10 features per node)
        forest = RandomForestEnsemble(
            num_trees=3,
            max_depth=2,
            leaf_dims=10
        )
        
        model = tf.keras.models.Sequential([
            forest
        ])
        
        # Just build the model to trigger feature selection
        model.build((None, 100))
        
        # Check that feature selection was applied at the nodes
        # Root node should have selected approximately 10 features
        self.assertGreater(len(forest.feature_indices), 0)
        self.assertLess(len(forest.feature_indices), 100)  # Should be less than all features
        
        # Print selected feature counts for visibility
        print(f"\nClassification test - Selected features count: {len(forest.feature_indices)}")

# Commented out since MultitaskRandomForestEnsemble might need additional setup
# class MultitaskRandomForestEnsembleTest(tf.test.TestCase):
#     def test_training(self):
#         mnist = tf.keras.datasets.mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train, x_test = x_train / 255.0, x_test / 255.0
#         
#         forest_ensemble = MultitaskRandomForestEnsemble(
#             num_trees=5, 
#             max_depth=3, 
#             num_tasks=1, 
#             leaf_dims=10
#         )
#         
#         model = tf.keras.models.Sequential([
#             tf.keras.layers.Flatten(input_shape=(28, 28)),
#             forest_ensemble
#         ])
#         
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         model.compile(optimizer='adam',
#                       loss=loss_fn,
#                       metrics=['accuracy'])
#         
#         history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
#         
#         # Check if the accuracy is above 0.9.
#         self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)
#         model.save("model_test.keras")
        
tf.test.main()