# Quantum Capstone Spring 2024

# Neural Network Model with Neurotransmitter Dynamics

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

 The model is trained using a combination of backpropagation, weight updates, and iterative validation. Here's a detailed breakdown of how the training process works:

1. Loading and Separation of Data: The data is loaded or separated into training and validation sets. This involves splitting the available data into two parts, typically using a certain percentage for training and the remainder for validation.

2. Creation of Batches: The training data is segmented into smaller batches to facilitate the training process. Each batch contains a subset of the training data and is used for successive iterations during training. This is often done to ensure that the entire dataset doesn't need to be loaded into memory at once, especially in the case of large datasets.

3. Training Loop: The training process iterates over the batches of training data for a certain number of epochs (defined by the `num_epochs` hyperparameter). For each batch, the following steps are typically performed:
   - Model Processing: The model function is used to process the input and compute the activations.
   - Gradient Computation: The gradients of the loss function with respect to the model's parameters are computed using backpropagation. This involves calculating how the loss changes with respect to each parameter in the network.
   - Weight Updates: The weights of the neural network are updated based on these gradients and a specific optimizer (in this case, the Adam optimizer). The Adam optimizer is a popular method for adapting the learning rate for each parameter.

4. Evaluation on Validation Set: After training on each batch, the model's performance is evaluated using the validation set. This is to ensure that the model is generalizing well and not overfitting to the training data.

5. Logging and Monitoring: Metrics such as validation loss are often logged and monitored during the training process to track the model's performance.

6. Main Loop: After the initial training loop, the model continues to iterate through a main loop, updating neurotransmitter levels, computing neural network activations, and performing weight updates using the Adam optimizer.

The entire process involves a combination of forward and backward passes through the neural network, utilizing the provided data, and iteratively adjusting model parameters to minimize the loss function and improve overall performance.

This comprehensive training process aims to optimize the model's performance and enable it to make accurate predictions or classifications based on the input data.

You can modify these parameters according to your requirements or experiment with different values to observe their impact on the model's performance.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project was developed as a capstone project for "The Coding School," from Quibit by Qubit, a Quantum research organization partnered with Google. Special thanks to my family for their guidance and support throughout the project.
```

This README.md file provides an overview of the project, instructions for getting started, configuration details, contributing guidelines, license information, and acknowledgments. You can customize the content as needed, such as adding more detailed installation instructions, project-specific dependencies, or any additional sections you find relevant.
