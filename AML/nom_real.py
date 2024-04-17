import tensorflow as tf
#from sklearn.preprocessing import StandardScaler
import numpy as np
import neural_optimization_machine as nom
import process_jld2 
import visualization as vs

# Define the neural network structure
def objective_function_model(X):

    N_features = X.shape[-1]
    input_shape = (N_features,)

    model = tf.keras.Sequential([
        nom.PolynomialFeatures(input_shape=input_shape),
        nom.StandardScaler(),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(128, activation='relu'),
        #tf.keras.layers.BatchNormalization(),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)  # Output layer
    ])

    polynomial_X = model.layers[0](X)
    model.layers[1].fit_data(polynomial_X)

    return model

data = process_jld2.get_data("./nom_data_2024-04-16_01:54:25.jld2", problemtype="results3", N_freq=2)
#data = process_jld2.get_data("./nom_combined.jld2", problemtype="results3", N_freq=3)

X, y = process_jld2.get_x_y(data)
y = np.log10(y) # Infidelity varies on exponential scale, need to scale down
#y = np.log10(y)**2 # Infidelity varies on exponential scale, need to scale down
#y = 1 / y # Infidelity varies on exponential scale, need to scale down

N_features = X.shape[-1]
N_instances = X.shape[0]
test_cutoff = (N_instances*4) // 5 # Keep 20 percent of data for testing

X_train = X[:test_cutoff]
y_train = y[:test_cutoff]

X_test = X[test_cutoff:]
y_test = y[test_cutoff:]


objective_model = objective_function_model(X)
objective_model.compile(optimizer='adam', loss='mse', metrics = [tf.keras.metrics.MeanAbsolutePercentageError()])

history = objective_model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2)

fig_train = vs.visualize_2freq(X_train, y_train, X_train, objective_model(X_train))
fig_test = vs.visualize_2freq(X_test, y_test, X_test, objective_model(X_test))
