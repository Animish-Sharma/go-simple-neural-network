# Simple Neural Network in Go

This project implements a simple **Feedforward Neural Network** from scratch in Go (Golang). It demonstrates the core concepts of machine learning—such as matrix multiplication, activation functions, and backpropagation—without relying on high-level ML frameworks (like TensorFlow or PyTorch). The only external dependency is [Gonum](https://www.gonum.org/) for efficient matrix operations.

The network is trained on the classic **Iris flower dataset** to classify iris flowers into three species based on four measurements (sepal length, sepal width, petal length, petal width).

## Features

- **From Scratch Implementation**: Understand the math behind neural networks.
- **Configurable Architecture**: Easily adjust the number of input, hidden, and output neurons.
- **Matrix Operations**: Uses `gonum/mat` for efficient linear algebra.
- **Sigmoid Activation**: Implements the sigmoid function and its derivative.
- **Backpropagation**: clear implementation of the gradient descent algorithm.

##  Prerequisites

- **Go**: Version 1.24 or later.
- **Git**: To clone the repository.

##  Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Animish-Sharma/go-simple-neural-network.git
    cd go-simple-neural-network
    ```

2.  **Install dependencies**:
    This project uses `gonum` for matrix math.
    ```bash
    go mod tidy
    ```

##  Usage

Run the program using the Go command line:

```bash
go run .
```

The program will:
1.  Load training data from `data/train.csv`.
2.  Train the neural network for a specified number of epochs.
3.  Load testing data from `data/test.csv`.
4.  Predict the species for the test data.
5.  Calculate and print the **accuracy** of the model.

**Expected Output:**
```text
Accuracy = 0.97
```
*(Accuracy may vary slightly due to random weight initialization)*

##  Code Walkthrough & Explanation

The core logic resides in `main.go`. Here is a detailed breakdown of each component and function.

### 1. Configuration (`neuralNetworkConfig`)

This struct holds the hyperparameters for the network.
```go
type neuralNetworkConfig struct {
    inputNeurons  int     // Number of input features (4 for Iris)
    outputNeurons int     // Number of classes (3 for Iris)
    hiddenNeurons int     // Number of neurons in the hidden layer
    numEpochs     int     // Number of training iterations
    learningRate  float64 // Step size for weight updates
}
```

### 2. Network Structure (`neuralNetwork`)

The network consists of weights (`w`) and biases (`b`) connecting the layers.
-   `wHidden`: Weights between Input and Hidden layer.
-   `bHidden`: Biases for the Hidden layer.
-   `wOut`: Weights between Hidden and Output layer.
-   `bOut`: Biases for the Output layer.

### 3. Initialization (`newNetwork`)

`newNetwork` initializes the struct. Note that the actual weights are initialized randomly at the start of the `train` function to break symmetry and ensure the network learns effectively.

### 4. Training (`train`)

This is where the magic happens.
1.  **Random Initialization**: Weights are filled with random float64 values.
2.  **Epoch Loop**: The training process repeats for `config.numEpochs`.
3.  **Backpropagation**: Inside the loop, `nn.backpropagate` is called to calculate errors and update weights.
4.  **Model Saving**: After training, the optimized weights and biases are stored in the `neuralNetwork` struct for future predictions.

### 5. Backpropagation (`backpropagate`)

This function performs one full iteration of **Forward Pass** followed by **Backward Pass**.

**Phase A: Forward Pass**
Calculates the network's output based on current weights.
1.  **Input -> Hidden**:
    $Input_{hidden} = X \cdot W_{hidden} + b_{hidden}$
    $Activation_{hidden} = Sigmoid(Input_{hidden})$
2.  **Hidden -> Output**:
    $Input_{output} = Activation_{hidden} \cdot W_{out} + b_{out}$
    $Output = Sigmoid(Input_{output})$

**Phase B: Backward Pass (Gradient Descent)**
Calculates how much each weight contributed to the error and adjusts them.
1.  **Error Calculation**: $Error = Target (Y) - Output$
2.  **Output Gradient**: Calculates the slope of the sigmoid function at the output layer ($Error \times Sigmoid'(Output)$).
3.  **Hidden Gradient**: Propagates the error backwards to the hidden layer using the transpose of weights ($W_{out}^T$).
4.  **Weight Updates**:
    -   Adjusts `wOut` and `bOut` using the Output Gradient.
    -   Adjusts `wHidden` and `bHidden` using the Hidden Gradient.
    -   The `learningRate` scales the size of these adjustments.

### 6. Prediction (`predict`)

Performs a **Forward Pass** only, using the trained weights to generate outputs for new, unseen data. It returns the probability distribution across the 3 classes.

### 7. Helper Functions

-   `sigmoid(x)`: The activation function $f(x) = \frac{1}{1 + e^{-x}}$. It squashes values between 0 and 1.
-   `sigmoidPrime(x)`: The derivative of the sigmoid function, used for training: $f'(x) = f(x) \cdot (1 - f(x))$.
-   `makeInputandLabels(filename)`: Parses the CSV file.
    -   First 4 columns -> Input Matrix (Features).
    -   Last 3 columns -> Label Matrix (One-hot encoded targets, e.g., `1.0, 0.0, 0.0`).
-   `sumAlongAxis(axis, m)`: A helper to sum matrix elements, used during bias updates.

##  Data Format

The CSV files (`train.csv`, `test.csv`) follow this format:
```csv
sepal_length,sepal_width,petal_length,petal_width,setosa,virginica,versicolor
0.08,0.66,0.0,0.04,1.0,0.0,0.0
...
```
-   **Features**: Normalized values for the 4 flower measurements.
-   **Labels**: One-hot encoded (1.0 indicates membership of that class).

##  Contribution

Feel free to fork this repository and experiment! fast-forwarding the learning rate or changing the number of hidden neurons is a great way to see how hyperparameters affect training.
