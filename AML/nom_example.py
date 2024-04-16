import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network structure
def objective_function_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    return model


def objective_function_analytic(x):
    return (x-2)**2

N_samples = 1000


np.random.seed(0)  # For reproducibility
x_train = np.random.uniform(low=-10, high=10, size=(N_samples, 1))  # Generate 1000 data points
y_train = (x_train - 2) ** 2  # Compute y for each x

x_test = np.random.uniform(low=-10, high=10, size=(100, 1))  # Generate 1000 data points
y_test = (x_test - 2) ** 2

# Add noise
y_train += np.random.normal(loc=0.0, scale=0.1, size=(N_samples, 1))




# Create the model
objective_model = objective_function_model()

# Compile the model
objective_model.compile(optimizer='adam', loss='mse')

# Train the model
objective_model.fit(x_train, y_train, epochs=5, batch_size=32)

objective_model.save_weights('my_weights.weights.h5')


plt.scatter(x_train, y_train, label="training data")
plt.scatter(x_test, y_test, label="test data")

y_pred = objective_model.predict(x_test)
plt.scatter(x_test, y_pred, label="NN pred")
plt.legend()

plt.show()

"""
A layer that does nothing but add biases to the input. In this way, we 
"""
class StartingPointLayer(tf.keras.layers.Layer):
    """
    Use base class constructor
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="kernel",
        )

        self.bias = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    """
    Apply weights elementwise to inputs, and add biases
    """
    def call(self, inputs):
        return tf.multiply(inputs, self.kernel) + self.bias


nom_layer = NomLayer(input_shape=(1,))

for layer in objective_model.layers:
    layer.trainable = False

print("Weights before optimization")
print(objective_model.get_weights())

outer_model = tf.keras.models.Sequential([
    nom_layer,
    objective_model
])


def identity_loss(y_true, y_pred):
    return y_pred

outer_model.compile(optimizer='adam', loss=identity_loss)

initial_guess = np.ones(1)
dummy_label = np.ones(1)

outer_model.fit(initial_guess, dummy_label, epochs=5, batch_size=1)

print("Weights after optimization")
print(objective_model.get_weights())


