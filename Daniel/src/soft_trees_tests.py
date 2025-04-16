import tensorflow as tf
from soft_trees import TreeEnsemble, MultitaskTreeEnsemble
import pandas as pd
import os

class TreeEnsembleTest(tf.test.TestCase):
    def test_training(self):
        #mnist = tf.keras.datasets.mnist
        #change to breast_cancer csv in Main_Implementation/pmlb/breast_cancer.csv
        #print current directory
        print(os.getcwd())
        #get absolute path to Main_Implementation
        main_implementation_path = os.path.abspath("../Main_Implementation")
        df = pd.read_csv(os.path.join(main_implementation_path, "pmlb_datasets/breast_cancer_wisconsin_original.csv"))
        x_train = df.iloc[:, :-1]
        y_train = df.iloc[:, -1]
        x_test = x_train
        y_test = y_train
        #(x_train, y_train), (x_test, y_test) = mnist.load_data()
        #_train, x_test = x_train / 255.0, x_test / 255.0
        # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
        # The tree ensemble will be used as the output layer ==> Each entry in the output vector
        # corresponds to one of the 10 digits.
        tree_ensemble = TreeEnsemble(5, 3, 2)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(30,)),
          tree_ensemble
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.summary()
        for l in model.layers:
            print("===============================================", l.name)
            for w in l.get_weights():
                print(w.shape)
        history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
        # Check if the accuracy is above 0.9.
        self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)

        
# class MultitaskTreeEnsembleTest(tf.test.TestCase):
#     def test_training(self):
#         mnist = tf.keras.datasets.mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train, x_test = x_train / 255.0, x_test / 255.0
#         # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
#         # The tree ensemble will be used as the output layer ==> Each entry in the output vector
#         # corresponds to one of the 10 digits.
#         tree_ensemble = MultitaskTreeEnsemble(5, 3, 1, 10)
#         model = tf.keras.models.Sequential([
#           tf.keras.layers.Flatten(input_shape=(28, 28)),
#           tree_ensemble
#         ])
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         model.compile(optimizer='adam',
#                       loss=loss_fn,
#                       metrics=['accuracy'])
#         history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
#         # Check if the accuracy is above 0.9.
#         self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)
#         model.save("model_test.keras")
        
tf.test.main()