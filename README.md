# Quantum_Capstone_Spring2024

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

Sure, here's a README.md file for your capstone project:

```markdown
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

   ```
   git clone https://github.com/your-username/neural-network-neurotransmitter-model.git
   ```

2. Install the required dependencies:

   ```
   pip install numpy
   ```

3. Run the main script:

   ```
   python main.py
   ```

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

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project was developed as a capstone project for [Your School/Program Name]. Special thanks to [Your Supervisor/Instructor Name] for their guidance and support throughout the project.
```

This README.md file provides an overview of the project, instructions for getting started, configuration details, contributing guidelines, license information, and acknowledgments. You can customize the content as needed, such as adding more detailed installation instructions, project-specific dependencies, or any additional sections you find relevant.
