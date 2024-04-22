import numpy as np
from sklearn.model_selection import train_test_split
import cirq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def apply_activation(x, activation_function):
    if activation_function == 'relu':
        return relu(x)
    elif activation_function == 'sigmoid':
        return sigmoid(x)
    elif activation_function == 'tanh':
        return tanh(x)
    # Add more activation functions as needed

# Hyperparameters
num_neurons = 100
num_layers = 3
hidden_activation_function = 'relu'
output_activation_function = 'sigmoid'
regularization_rate = 0.01
optimizer = 'adam'
learning_rate = 0.001
beta1 = 0.9  # Adam params
beta2 = 0.999
epsilon = 1e-8
num_epochs = 100
batch_size = 32
num_features = 300 

# Define neurotransmitter types and initial conditions
neurotransmitters = ["Dopamine", "Serotonin", "GABA", "Glutamate", "Acetylcholine"]
initial_neurotransmitter_levels = [0.5, 0.3, 0.2, 0.4, 0.1]

# Initialize state vectors
position_states = np.zeros((num_neurons, 300), dtype=complex)
neurotransmitter_states = np.zeros((num_neurons, len(neurotransmitters)), dtype=float)
neurotransmitter_states += initial_neurotransmitter_levels

# Set initial position (example - neuron 5 with 100% probability)
position_states[4, 1] = 1

# Coin state for superposition during movement
coin_state = np.array([1, 1j]) / np.sqrt(2)

# Helper functions
def l2_regularization(weights, regularization_rate):
    return regularization_rate * np.sum(np.square(weights))

def release_neurotransmitter(neuron_index, neurotransmitter_type, release_probability):
    if np.random.random() < release_probability:
        neurotransmitter_states[neuron_index, neurotransmitter_type] += 1

def reuptake_neurotransmitter(neuron_index, neurotransmitter_type, reuptake_rate):
    neurotransmitter_states[neuron_index, neurotransmitter_type] -= reuptake_rate

def interact_with_neurotransmitters(position_states, neurotransmitter_states):
    updated_position_states = np.copy(position_states)
    for neuron_index in range(num_neurons):
        neurotransmitter_effects = np.zeros(len(neurotransmitters))
        for nt_index, nt_type in enumerate(neurotransmitters):
            if nt_type == "Dopamine":
                neurotransmitter_effects[nt_index] = 0.2
            elif nt_type == "GABA":
                neurotransmitter_effects[nt_index] = -0.1

        combined_effect = np.dot(neurotransmitter_states[neuron_index], neurotransmitter_effects)
        updated_position_states[neuron_index] += combined_effect * updated_position_states[neuron_index]

    updated_position_states /= np.linalg.norm(updated_position_states, axis=1)[:, None]
    return updated_position_states

# Model function
def model(position_states_batch, neurotransmitter_states_batch, weights):
    input_states = position_states_batch
    
    layer_outputs = [input_states]
    for layer_idx in range(num_layers - 1):
        weights_idx = layer_idx * 2
        biases_idx = weights_idx + 1
        layer_input = layer_outputs[-1]
        layer_weights = weights[weights_idx]
        layer_biases = weights[biases_idx]
        layer_output = np.dot(layer_input, layer_weights.T) + layer_biases
        layer_outputs.append(apply_activation(layer_output, hidden_activation_function))
    
    output_layer_output = np.dot(layer_outputs[-1], weights[-2].T) + weights[-1]
    output_activation = apply_activation(output_layer_output, output_activation_function)
    
    activations = output_activation
    
    regularization_loss = 0
    for layer_idx in range(num_layers):
        weights_idx = layer_idx * 2
        layer_weights = weights[weights_idx]
        regularization_loss += l2_regularization(layer_weights, regularization_rate)
    
    position_states_batch = interact_with_neurotransmitters(position_states_batch, neurotransmitter_states_batch)
    
    return position_states_batch, activations, regularization_loss, layer_outputs

# Weight initialization
def xavier_initialization(input_size, output_size):
    xavier_stddev = np.sqrt(2.0 / (input_size + output_size))
    return np.random.normal(0, xavier_stddev, size=(input_size, output_size))

input_size = num_features
hidden_size = 300
output_size = num_features

weights = []
weights.append(xavier_initialization(input_size, hidden_size))
weights.append(np.zeros(hidden_size))

for i in range(num_layers - 2):
    weights.append(xavier_initialization(hidden_size, hidden_size))
    weights.append(np.zeros(hidden_size))

weights.append(xavier_initialization(hidden_size, output_size))
weights.append(np.zeros(output_size))

# Adam optimizer
def update_weights_adam(weights, grads, m, v):
    updated_weights = []
    updated_m = []
    updated_v = []

    for i, (w, g, m_i, v_i) in enumerate(zip(weights, grads, m, v)):
        m_i = beta1 * m_i + (1 - beta1) * g
        v_i = beta2 * v_i + (1 - beta2) * g**2
        m_hat = m_i / (1 - beta1**(i + 1))
        v_hat = v_i / (1 - beta2**(i + 1))
        updated_weights.append(w - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
        updated_m.append(m_i)
        updated_v.append(v_i)

    return updated_weights, updated_m, updated_v

# Compute gradients
def compute_gradients(activations, targets, weights, layer_outputs):
    # Compute the gradients using backpropagation
    num_samples = activations.shape[0]

    # Compute the loss
    loss = np.mean((activations - targets)**2)

    # Compute the gradients
    output_gradient = 2 * (activations - targets) / num_samples
    input_gradient = np.dot(output_gradient, weights[-2])
    weights_gradient = np.dot(layer_outputs[-2].T, output_gradient)
    biases_gradient = np.sum(output_gradient, axis=0)

    # Store the gradients
    grads = [weights_gradient, biases_gradient]

    return grads, loss

# Generate synthetic data for testing
num_samples = 1000
num_features = 300
X_data = np.random.rand(num_samples, num_features)
y_data = np.random.randint(0, 2, size=(num_samples, num_features))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Training loop
m = [np.zeros_like(w) for w in weights]
v = [np.zeros_like(w) for w in weights]

for epoch in range(num_epochs):
    # Iterate over batches
    for batch_idx in range(0,len(X_train), batch_size):
        batch_position_states = X_train[batch_idx:batch_idx+batch_size]
        batch_neurotransmitter_states = neurotransmitter_states[batch_idx:batch_idx+batch_size]
        
        position_states_batch, activations, regularization_loss, layer_outputs = model(batch_position_states, batch_neurotransmitter_states, weights)
        
        grads, loss = compute_gradients(activations, y_train[batch_idx:batch_idx+batch_size], weights, layer_outputs)

        for i, (w, g, m_i, v_i) in enumerate(zip(weights, grads, m, v)):
            m_i = beta1 * m_i + (1 - beta1) * g
            v_i = beta2 * v_i + (1 - beta2) * g**2
            m_hat = m_i / (1 - beta1**(epoch + 1))
            v_hat = v_i / (1 - beta2**(epoch + 1))
            weights[i] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    # Evaluate on validation set
    val_loss = 0
    for val_batch_idx in range(0, len(X_val), batch_size):
        val_batch_position_states = position_states[val_batch_idx:val_batch_idx+batch_size]
        val_batch_neurotransmitter_states = neurotransmitter_states[val_batch_idx:val_batch_idx+batch_size]
        val_batch_weights = weights[val_batch_idx:val_batch_idx+batch_size]

        _, val_activations, _, val_layer_outputs = model(val_batch_position_states, val_batch_neurotransmitter_states, val_batch_weights)
        _, batch_loss = compute_gradients(val_activations, y_val[val_batch_idx:val_batch_idx+batch_size], weights, val_layer_outputs)
        val_loss += batch_loss

    val_loss /= (len(X_val) // batch_size)

    # Print/log metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

# Main loop
steps = 0
while True:
    for neuron_index in range(num_neurons):
        for nt_index, nt_type in enumerate(neurotransmitters):
            release_probability = 0.1 * np.exp(-steps / 10)
            reuptake_rate = 0.05
            release_neurotransmitter(neuron_index, nt_index, release_probability)
            reuptake_neurotransmitter(neuron_index, nt_index, reuptake_rate)

    position_states, activations, regularization_loss, layer_outputs = model(position_states, neurotransmitter_states, weights)

    if optimizer == 'adam':
        weights, m, v = update_weights_adam(weights, grads, m, v)

    steps += 1

    # Quantum Simulation
    # Create quantum states
    zero_state = np.array([1, 0])
    one_state = np.array([0, 1])
    zero_one_state = zero_state + one_state

    # Normalize the zero_one_state
    length = np.sqrt(np.dot(zero_one_state, zero_one_state))
    zero_one_state = zero_one_state / length

    # Define quantum gates
    id_gate = np.array([[1, 0], [0, 1]])
    x_gate = np.array([[0, 1], [1, 0]])
    z_gate = np.array([[1, 0], [0, -1]])
    h_gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])

    # Apply a gate to the zero_one_state
    state = np.dot(z_gate, zero_one_state)
    probabilities = state**2

    # Simulate measurements
    num_measurements = 500
    counts = np.random.multinomial(num_measurements, probabilities)

    # Plot measurements histogram
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.title('Histogram of Measurement Results (Zero-One State)')
    plt.show()

    # Simulate a rotational Y gate
    theta = np.radians(30)
    ry_gate = np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ])

    state = zero_state
    circuit = [ry_gate]

    for gate in circuit:
        state = np.dot(gate, state)

    probabilities = state**2

    # Simulate measurements
    counts = np.random.multinomial(num_measurements, probabilities)

    # Plot measurements histogram
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.title('Histogram of Measurement Results (Rotational Y Gate)')
    plt.show()

    # Visualization with Grok sphere
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot input features as points on the x-axis
    x = X_data
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    ax.scatter(x, y, z, c='b', label='Input Features')

    # Plot target output as points on the y-axis
    x = np.zeros_like(y_data)
    y = y_data
    z = np.zeros_like(y_data)
    ax.scatter(x, y, z, c='g', label='Target Output')

    # Plot predicted probabilities as points on the z-axis
    x = np.zeros_like(activations)
    y = np.zeros_like(activations)
    z = activations
    ax.scatter(x, y, z, c='r', label='Predicted Probabilities')

    ax.set_xlabel('Input Features')
    ax.set_ylabel('Target Output')
    ax.set_zlabel('Predicted Probabilities')
    ax.legend()

    plt.show()
    