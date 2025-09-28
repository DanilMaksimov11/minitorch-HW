"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""
import random

import minitorch


class Network(minitorch.Module):
    """A simple neural network with three linear layers and ReLU activations.

    Attributes:
        layer1 (Linear): First linear layer (input to hidden).
        layer2 (Linear): Second linear layer (hidden to hidden).
        layer3 (Linear): Third linear layer (hidden to output).
    """

    def __init__(self, hidden_layers):
        """Initializes the Network with three linear layers.

        Args:
            hidden_layers (int): Number of neurons in each hidden layer.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """Performs forward pass through the network.

        Args:
            x (tuple): Input tuple of two Scalars.

        Returns:
            minitorch.Scalar: Output after applying sigmoid activation.
        """
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()


class Linear(minitorch.Module):
     """A linear layer implementing y = xW + b.

    Attributes:
        weights (list): 2D list of weight parameters.
        bias (list): List of bias parameters.
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []
        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    def forward(self, inputs):
        """Performs forward pass of the linear layer.

        Args:
            inputs (list): List of input Scalars.

        Returns:
            list: Output list of Scalars after linear transformation.
        """
        y = [b.value for b in self.bias]
        for i, x in enumerate(inputs):
            for j in range(len(y)):
                y[j] = y[j] + x * self.weights[i][j].value
        return y
        
def default_log_fn(epoch, total_loss, correct, losses):
    """Default logging function that prints training progress.

    Args:
        epoch (int): Current epoch number.
        total_loss (float): Total loss for the epoch.
        correct (int): Number of correct predictions.
        losses (list): List of loss values for all epochs.
    """
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    """Training class for scalar-based network using SGD."""

    def __init__(self, hidden_layers):
        """Initializes trainer with network architecture.

        Args:
            hidden_layers (int): Number of neurons in hidden layers.
        """
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        """Runs a single forward pass for input x.

        Args:
            x (tuple): Input tuple of two numerical values.

        Returns:
            minitorch.Scalar: Network output after sigmoid activation.
        """
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        """Trains the model using stochastic gradient descent.

        Args:
            data: Training data with X and y attributes.
            learning_rate (float): SGD learning rate.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 500.
            log_fn (callable, optional): Logging function. Defaults to default_log_fn.
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    ScalarTrain(HIDDEN).train(data, RATE)
