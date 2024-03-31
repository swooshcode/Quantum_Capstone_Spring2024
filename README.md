# Quantum Capstone Spring 2024

# Quantum Neural Network Model with Neurotransmitter Dynamics

This repository contains the implementation of a neural network model that incorporates neurotransmitter dynamics and interactions. The model simulates the release, reuptake, and effects of various neurotransmitters on neuronal positions and movement probabilities.

## Overview

The model consists of the following components:

- Multi-layer neural network with configurable number of layers and neurons
- Simulation of neurotransmitter release and reuptake mechanisms
- Modeling of excitatory and inhibitory effects of neurotransmitters on neuronal movement
- Integration of neurotransmitter interactions with neural network computations
- Weight initialization using Xavier initialization
- Training loop with Adam optimizer and L2 regularization
- Evaluation on a validation set and logging of training metrics

The model is implemented in Python using NumPy for numerical computations.

## Changes

### Update 1: Optimization Techniques

Several optimization techniques have been added to enhance the network architecture, loss function, and hyperparameters for real-world scenarios. These include:

- Experimenting with different numbers of layers and neurons
- Considering convolutional and recurrent layers for specific tasks
- Investigating attention mechanisms and different activation functions
- Choosing appropriate loss functions and regularization techniques
- Performing hyperparameter tuning
- Investigating quantum algorithms and quantum circuit integration
- Applying data preprocessing and augmentation techniques
- Using appropriate evaluation metrics and validation strategies
- Continuously monitoring and refining the model based on real-world performance

### Update 2: Training Process

The model is trained using a combination of backpropagation, weight updates, and iterative validation. The training process includes:

1. Loading and separation of data into training and validation sets
2. Creation of batches for efficient training
3. Training loop with gradient computation and weight updates using the Adam optimizer
4. Evaluation on the validation set to assess generalization performance
5. Logging and monitoring of training metrics
6. Main loop for updating neurotransmitter levels and performing neural network computations

## Getting Started

To run the model, follow these steps:

1. Clone the repository:
git clone https://github.com/your-username/neural-network-neurotransmitter-model.git

2. Install the required dependencies:
pip install numpy

3. Run the main script:
python main.py

This will execute the model, including the training loop and the main loop for simulating neurotransmitter dynamics.

## Configuration

The model's hyperparameters and settings can be configured in the `main.py` file. Some of the key parameters include:

- `num_neurons`: Number of neurons in the neural network
- `num_layers`: Number of layers in the neural network
- `activation_function`: Activation function to be used (e.g., 'relu', 'sigmoid')
- `regularization_rate`: L2 regularization rate
- `optimizer`: Optimizer to be used (e.g., 'adam', 'sgd')
- `learning_rate`: Learning rate for the optimizer
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training

You can modify these parameters according to your requirements or experiment with different values to observe their impact on the model's performance.

## Ways to Improve

Here are some ways that you can improve the code:

1. Change the values of these variables based on real-world scenarios:
```
num_samples, num_features, X_data, y_data.
```
2. Implement additional activation functions and experiment with their impact on the model's performance.
3. Explore different network architectures, such as convolutional or recurrent layers, depending on the specific task and data characteristics.
4. Investigate the integration of quantum algorithms and quantum circuits to leverage quantum computing capabilities.
5. Implement data preprocessing and augmentation techniques to enhance the model's robustness and generalization ability.
6. Conduct extensive hyperparameter tuning to find the optimal combination of hyperparameters for the specific problem.
7. Incorporate additional evaluation metrics and validation strategies to assess the model's performance comprehensively.
8. Continuously monitor and refine the model based on real-world performance and user feedback.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project was developed as a capstone project for "The Coding School," from Quibit by Qubit, a Quantum research organization partnered with Google. Special thanks to my family and friends for their guidance and support throughout the project.
