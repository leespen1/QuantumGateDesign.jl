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

# Create the model
objective_model = objective_function_model()

# Compile the model
objective_model.compile(optimizer='adam', loss='mse')


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
objective_model.fit(x_train, y_train, epochs=50, batch_size=32)


plt.scatter(x_train, y_train, label="training data")
plt.scatter(x_test, y_test, label="test data")

y_pred = objective_model.predict(x_test)
plt.scatter(x_test, y_pred, label="NN pred")

plt.show()
