import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import process_jld2 
# Define the neural network structure
def objective_function_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    return model

#data = process_jld2.get_data("./nom_data_2024-04-16_01:54:25.jld2", problemtype="results3", N_freq=3)
data = process_jld2.get_data("./nom_combined.jld2", problemtype="results3", N_freq=3)
X, y = process_jld2.get_x_y(data)
scaler = StandardScaler()

X = scaler.fit_transform(X)
y = np.log10(y) # Infidelity varies on exponential scale, need to scale down

N_features = X.shape[-1]
N_instances = X.shape[0]
test_cutoff = (N_instances*4) // 5 # Keep 20 percent of data for testing

X_train = X[:test_cutoff]
y_train = y[:test_cutoff]

X_test = X[test_cutoff:]
y_test = y[test_cutoff:]


objective_model = objective_function_model(input_shape=(N_features,) )
objective_model.compile(optimizer='adam', loss='mse')

history = objective_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)
