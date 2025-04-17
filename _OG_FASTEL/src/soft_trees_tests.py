import tensorflow as tf
from soft_trees import TreeEnsemble, MultitaskTreeEnsemble
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from data_utils import load_processed_classification_public_data, get_processed_data

def get_df(name = "breast-cancer-wisconsin"):
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    #get absolute path to Main_Implementation
    main_implementation_path = os.path.abspath("../Main_Implementation")
    full_path = os.path.join(main_implementation_path, "pmlb_datasets/breast_cancer_wisconsin_original.csv")
    print(full_path)
    df = pd.read_csv(full_path)
    return df

class TreeEnsembleTest(tf.test.TestCase):
    def test_training(self):
        #mnist = tf.keras.datasets.mnist
        #change to breast_cancer csv in Main_Implementation/pmlb/breast_cancer.csv
        #print directory of this file
        #set current directory to parent directory of this file
        data = load_processed_classification_public_data(name = 'breast-cancer-wisconsin')
        #split train and test into random 70/30
        x_train = data.x_train_processed
        y_train = data.y_train_processed
        x_test = data.x_test_processed
        y_test = data.y_test_processed

        #(x_train, y_train), (x_test, y_test) = mnist.load_data()
        #_train, x_test = x_train / 255.0, x_test / 255.0
        # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
        # The tree ensemble will be used as the output layer ==> Each entry in the output vector
        # corresponds to one of the 10 digits.
        tree_ensemble = TreeEnsemble(10, 3, 2)
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
          tree_ensemble
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.summary()
        for l in model.layers:
            print("===============================================", l.name)
            # for w in l.get_weights():
            #     print(w.shape)
        history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
        # Check if the accuracy is above 0.9.
        #self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)
        print(str(history.history))

        
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

   