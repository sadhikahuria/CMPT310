import math
import nn
import numpy as np


import util
###########################################################################
class NaiveBayesDigitClassificationModel(object):

    def __init__(self):
        self.conditionalProb = None
        self.prior = None
        self.features = None
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.legalLabels = range(10)

    def train(self, dataset):
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in dataset.trainingData for f in datum.keys()]))

        kgrid = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.5, 1, 5]
        self.trainAndTune(dataset, kgrid)

    def trainAndTune(self, dataset, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        trainingData = dataset.trainingData
        trainingLabels = dataset.trainingLabels
        validationData = dataset.validationData
        validationLabels = dataset.validationLabels

        bestAccuracyCount = -1  # best accuracy so far on validation set
        #Common training - get all counts from training data
        #We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter()  # Prior probability over labels
        commonConditionalProb = util.Counter()  #Conditional probability of feature feat being 1 indexed by (feat, label)
        commonCounts = util.Counter()  #how many time I have seen feature 'feat' with label 'y' whether inactive or active
        bestParams = (commonPrior, commonConditionalProb, kgrid[0])  # used for smoothing part  trying various Laplace factors kgrid

        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = int(trainingLabels[i])
            "*** YOUR CODE HERE to complete populating commonPrior, commonCounts, and commonConditionalProb ***"
            commonPrior[label] += 1
            for feat in self.features:
                commonCounts[(feat, label)] += 1
                if datum[feat] == 1:
                    commonConditionalProb[(feat, label)] += 1
                    

        for k in kgrid:  # smoothing parameter tuning loop
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in commonPrior.items():
                prior[key] += val
            for key, val in commonCounts.items():
                counts[key] += val
            for key, val in commonConditionalProb.items():
                conditionalProb[key] += val

            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    "*** YOUR CODE HERE to update conditionalProb and counts using Lablace smoothing ***"
                    conditionalProb[(feat, label)] = (
                        conditionalProb[(feat, label)] + k
                    ) / (counts[(feat, label)] + 2 * k)

            # normalizing:
            prior.normalize()
            "**** YOUR CODE HERE to normalize conditionalProb "


            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation
            predictions = self.classify(validationData)
            accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print("Performance on validation set for k=%f: (%.1f%%)" % (
            k, 100.0 * accuracyCount / len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior, conditionalProb, k)
                bestAccuracyCount = accuracyCount
            # end of automatic tuning loop

        self.prior, self.conditionalProb, self.k = bestParams
        print("Best Performance on validation set for k=%f: (%.1f%%)" % (
            self.k, 100.0 * bestAccuracyCount / len(validationLabels)))


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            ("***YOUR CODE HERE***  use calculateLogJointProbabilities() to compute posterior per datum  and use"
             "it to find best guess digit for datum and at the end accumulate in self.posteriors for later use")
            
            logJoint = self.calculateLogJointProbabilities(datum)
            bestLabel = logJoint.argMax()
            guesses.append(bestLabel)
            self.posteriors.append(logJoint)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            "*** YOUR CODE HERE, to populate logJoint() list ***"
            prob = math.log(self.prior[label]) if self.prior[label] > 0 else float ('-inf')
            
            for feat in self.features:
                p = self.conditionalProb[(feat, label)]

                if datum[feat] == 1:
                    prob += math.log(p) if p > 0 else float('-inf')
                else:
                    prob += math.log(1 - p) if (1 - p) > 0 else float ('-inf')
            
            logJoint[label] = prob

        return logJoint

################################################################################3
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        return 1 if nn.as_scalar(score) >= 0 else -1
    

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            mistakes = 0

            for (x, y) in dataset.iterate_once(1):
                true_label = int(nn.as_scalar(y))
                prediction = self.get_prediction(x)
                if prediction != true_label:
                    mistakes += 1

                    self.w.update(x, true_label)
            if mistakes == 0: 
                break

########################################################################33
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here. Here you setup the architecture of your NN, meaning how many
        # layers and corresponding weights, what is the batch_size, and learning_rate.
        "*** YOUR CODE HERE ***"
        self.hidden_size = 100
        self.learning_rate = 0.001
        self.batch_size = 50
        self.max_epochs = 5000

        self.w1 = nn.Parameter(1, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)

        self.w2 = nn.Parameter(self.hidden_size, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        hidden_linear = nn.Linear(x, self.w1)
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)
        hidden = nn.ReLU(hidden_with_bias)

        output_linear = nn.Linear(hidden, self.w2)
        output_with_bias = nn.AddBias(output_linear, self.b2)
        return output_with_bias


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        loss = nn.SquareLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
            Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for epoch in range(self.max_epochs):
            for (x, y) in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)

                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])

                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)


##########################################################################
class DigitClassificationModel(object):
    """
    A second model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to classify each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 100
        self.learning_rate = 0.1
        self.batch_size = 100
        self.max_epochs = 20

        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
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
        "*** YOUR CODE HERE ***"
        hidden_linear = nn.Linear(x, self.w1)
        hidden_with_bias = nn.AddBias(hidden_linear, self.b1)
        hidden = nn.ReLU(hidden_with_bias)

        output_linear = nn.Linear(hidden, self.w2)
        logits = nn.AddBias(output_linear, self.b2)
        return logits

    def get_loss(self, x, y):
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
        "*** YOUR CODE HERE ***"
        logits = self.run(x)
        loss = nn.SoftmaxLoss(logits, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for epoch in range(self.max_epochs):
            for (x, y) in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.w2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)



###################################################################################
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 128
        self.learning_rate = 0.1
        self.batch_size = 100
        self.max_epochs = 10

        self.Wx = nn.Parameter(self.num_chars, self.hidden_size)
        self.Wh = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b = nn.Parameter(1, self.hidden_size)

        self.W_out = nn.Parameter(self.hidden_size, len(self.languages))
        self.b_out = nn.Parameter(1, len(self.languages))



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        batch_size = xs[0].data.shape[0]
        h = nn.Constant(np.zeros((batch_size, self.hidden_size)))

        for x in xs:
            input_contrib = nn.Linear(x, self.Wx)
            hidden_contrib = nn.Linear(h, self.Wh)

            h = nn.ReLU(nn.AddBias(nn.Add(input_contrib, hidden_contrib), self.b))

        logits = nn.AddBias(nn.Linear(h, self.W_out), self.b_out)
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(xs)
        loss = nn.SoftmaxLoss(logits, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        "*** hint: User get_validation_accuracy() to decide when to finish learning ***"
        for epoch in range(self.max_epochs):
            for (xs, y) in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.Wx, self.Wh, self.b, self.W_out, self.b_out])
                
                self.Wx.update(gradients[0], -self.learning_rate)
                self.Wh.update(gradients[1], -self.learning_rate)
                self.b.update(gradients[2], -self.learning_rate)
                self.W_out.update(gradients[3], -self.learning_rate)
                self.b_out.update(gradients[4], -self.learning_rate)
