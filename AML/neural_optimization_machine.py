import tensorflow as tf
from tensorflow.keras.initializers import Constant
from itertools import combinations_with_replacement
import numpy as np
"""
A layer that does nothing but apply weights and add biases to the input. In
this way, we obtain a 'controllable input'.
"""
class StartingPointLayer(tf.keras.layers.Layer):
    """
    Use base class constructor
    """
    def __init__(self, N_fixed_variables=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N_fixed_variables = N_fixed_variables

    def build(self, input_shape):
        self.tunable_weights = self.add_weight(
            shape=(input_shape[-1]-self.N_fixed_variables,),
            initializer="ones",
            trainable=True,
            name="tunable_weights",
        )


        self.tunable_bias = self.add_weight(
            shape=(input_shape[-1]-self.N_fixed_variables,),
            initializer="zeros",
            trainable=True,
            name="tunable_bias",
        )

    """
    Apply weights elementwise to (non-fixed) inputs, and add biases
    """
    def call(self, inputs):
        fixed_variables = inputs[:, :self.N_fixed_variables]
        design_variables = tf.multiply(inputs[:, self.N_fixed_variables:], self.tunable_weights) + self.tunable_bias
        return tf.concat([fixed_variables, design_variables], axis=-1)

"""
'Dummy' loss which does not depend on the actual label. Needed so we can
minimize the output of a model, in an unsupervised way (but tensorflow expects
a label, so we have to accept a y_label argument).
"""
def identity_loss(y_true, y_pred):
    return y_pred

"""
Create a neural optimization machine using an existing model. 
"""
def make_NeuralOptimizationMachine(obj_func_model, N_fixed_variables=0):
    # Create a layer which allows the input to the inner model to be determined
    # entirely by the weights and biases.
    starting_layer = StartingPointLayer(N_fixed_variables, input_shape=(obj_func_model.input_shape[-1],))

    # Clone the objective function model, copy weights
    obj_func_model_clone = tf.keras.models.clone_model(obj_func_model)
    obj_func_model_clone.set_weights(obj_func_model.get_weights())

    # objective function model should not be trainable when optimizing
    for layer in obj_func_model_clone.layers:
        layer.trainable = False
        # Also disable dropout
        if isinstance(layer, tf.keras.layers.Dropout):
            assert(hasattr(layer, 'rate')) # Make sure layer has rate
            layer.rate = 0

    # Connect input to the objective function model
    neural_optimization_model = tf.keras.models.Sequential([
        starting_layer,
        obj_func_model_clone,
    ])
    #neural_optimization_model = obj_func_model_clone(starting_layer, training=False)
    neural_optimization_model.compile(optimizer='adam', loss=identity_loss)

    return neural_optimization_model

def get_results_NeuralalOptimizationMachine(neural_optimization_model, x):
    optimal_x = neural_optimization_model.layers[0](x)
    loss = neural_optimization_model(x)
    return optimal_x, loss


"""
Because I need the input of the objective function model to be the frequencies
and the physical parameters, but the physical parameters need to be fixed in the
neural optimization machine, I need to write my own PolynomialFeatures layer so
I can generate the polynomial features.

Otherwise I would have, for example, x is constant, y is trainable, and xy is
trainable directly in a way so that it doesnt actually equal xy.
"""
class PolynomialFeatures(tf.keras.layers.Layer):
    def __init__(self, degree=2, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        input_dim = input_shape[-1]
        indices = []

        # Generate all combinations of feature indices for each degree
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(input_dim), d):
                indices.append(combo)

        self.indices = indices
        super(PolynomialFeatures, self).build(input_shape)

    def call(self, inputs):
        # Create the polynomial features
        outputs = []
        for index_tuple in self.indices:
            feature = tf.reduce_prod(tf.gather(inputs, index_tuple, axis=1), axis=1, keepdims=True)
            outputs.append(feature)

        return tf.concat(outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], len(self.indices))


class StandardScaler(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # These will hold arrays for the mean and weight of each feature
        self.mean = None
        self.std = None

    def build(self, input_shape):
        # Initialize the mean and std to 0 and 1 respectively
        self.mean = self.add_weight(name='mean',
                                    shape=(input_shape[-1],),
                                    initializer=Constant(0.0),
                                    trainable=False)
        self.std = self.add_weight(name='std',
                                   shape=(input_shape[-1],),
                                   initializer=Constant(1.0),
                                   trainable=False)
        super(StandardScaler, self).build(input_shape)

    """
    Set the mean and standard deviation to reflect a dataset.
    """
    def fit_data(self, data):
        # Compute the mean and std of the data
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)
        # Set the mean and std to the computed values
        self.mean.assign(mean)
        self.std.assign(std)

    def call(self, inputs):
        # Perform standardization
        return (inputs - self.mean) / (self.std + tf.keras.backend.epsilon())


def get_results_NeuralalOptimizationMachine(neural_optimization_model, x):
    optimal_x = neural_optimization_model.layers[0](x)
    loss = neural_optimization_model(x)
    return optimal_x, loss

"""
Breaks labels into bins, and resamples points from bins with fewer entries
until there is equal representation of points from each bin. This is done because
'good' carrier wave frequencies are exponentially rare, so we want to make sure
their impact is felt on the NN regressor (otherwise a low loss can be obtained
by just always predicting an infidelity of -5e-5, which is where most random
pulses end up)

BE CAREFUL TO ONLY RESAMPLE TRAINING DATA, SO WE DONT ACCIDENTALLY TRAIN ON TEST DATA
"""
def resample_bins(X, y, bins=np.array([-5, -4, -3, -2, -1, 0])):
    print(f"Using bins: {bins}")
    # Digitize the labels to find out which bin they belong to
    indices = np.digitize(y, bins, right=True)

    # Find the number of samples per bin
    num_samples_per_bin = np.array([np.sum(indices == i) for i in range(1, len(bins))])
    print(f"Number of samples in each bin: {num_samples_per_bin}")

    # Determine the maximum number of samples in any bin
    max_samples = np.max(num_samples_per_bin)

    # Initialize an array to hold the resampled indices
    resampled_indices = np.array([], dtype=int)

    # Resample each bin
    for i in range(1, len(bins)):
        # Get the indices of all samples in bin i
        bin_indices = np.where(indices == i)[0]
        # If this bin has fewer samples than max_samples, replicate them
        if len(bin_indices) < max_samples:
            # Calculate the number of times to repeat the bin to reach close to max_samples
            repeat_times = int(np.ceil(max_samples / len(bin_indices)))
            resampled_bin_indices = np.tile(bin_indices, repeat_times)[:max_samples]
        else:
            resampled_bin_indices = bin_indices
        # Append the resampled indices for this bin
        resampled_indices = np.concatenate((resampled_indices, resampled_bin_indices))

    # Shuffle resampled indices to randomize data order
    np.random.shuffle(resampled_indices)

    # Create the new resampled dataset
    X_resampled = X[resampled_indices]
    y_resampled = y[resampled_indices]

    return X_resampled, y_resampled

