import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        resultat = nn.as_scalar(self.run(x))
        if resultat >= 0:
            return 1
        else:
            return -1
    

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        converge = False
        while not converge:
            converge = True
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                actual = nn.as_scalar(y)
                # si un point et encore mal classé, on met à jour les poids et on continue l'entraînement
                if prediction != actual:
                    self.w.update(x, actual)
                    converge = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        self.w1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)
        self.w2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.w3 = nn.Parameter(64, 1)
        self.b3 = nn.Parameter(1, 1)
        self.lr = 0.01
        self.batch_size = 50

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, params)
                for param, grad in zip(params, grads):
                    param.update(grad, -self.lr)
            # Vérifier la convergence sur tout le dataset
            total_loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            if nn.as_scalar(total_loss) < 0.001:
                break


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        self.w1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.w2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.w3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)
        self.lr = 0.1
        self.batch_size = 100

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        h1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, self.w2), self.b2))
        output = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return output

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, params)
                for param, grad in zip(params, grads):
                    param.update(grad, -self.lr)
            accuracy = dataset.get_validation_accuracy()
            if accuracy >= 0.975:
                break
