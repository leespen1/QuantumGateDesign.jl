import tensorflow as tf

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




