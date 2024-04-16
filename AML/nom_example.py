import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import neural_optimization_machine as nom

# Define the neural network structure
def objective_function_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    return model


def objective_function_analytic(x):
    return (x[1]-2)**2 + x[0]

N_samples = 1000


np.random.seed(0)  # For reproducibility
x_train = np.random.uniform(low=-10, high=10, size=(N_samples, 2))  # Generate 1000 data points
#x_train[:,0] = 1

y_train = np.array([objective_function_analytic(x) for x in x_train])
y_train += np.random.normal(loc=0.0, scale=0.1, size=(N_samples,))

x_test = np.random.uniform(low=-10, high=10, size=(100, 2))  # Generate 1000 data points
x_test[:,0] = 1

y_test = np.array([objective_function_analytic(x) for x in x_test])
y_test += np.random.normal(loc=0.0, scale=1, size=(100,))



# Create the model
objective_model = objective_function_model(input_shape=(2,))
# Compile the model - mse minimizes least squares diffrence between labels and predicitons
objective_model.compile(optimizer='adam', loss='mse')

# Train the model
history = objective_model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

objective_model.save_weights('my_weights.weights.h5')


#plt.scatter(x_train[:,1], y_train, label="training data")
plt.scatter(x_test[:,1], y_test, label="test data")

y_pred = objective_model.predict(x_test)
plt.scatter(x_test[:,1], y_pred, label="NN pred")
plt.legend()

plt.show()

N_fixed_variables = 1
N_variables = x_train.shape[-1]

outer_model = nom.make_NeuralOptimizationMachine(objective_model, N_fixed_variables=1)


initial_guess = np.ones((1,2))
initial_guess[0,1] = -4
dummy_label = np.ones((1,1))

outer_model.fit(initial_guess, dummy_label, epochs=500, batch_size=1)

optimal_x, loss = nom.get_results_NeuralalOptimizationMachine(outer_model, initial_guess)
print(f"Found optimal parameters {optimal_x}\nWith expected loss {loss}")


