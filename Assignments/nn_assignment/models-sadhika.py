"""
Date: April 6 2025
Name: Sadhika Huria
COURSE: CMPT 310
Student ID: 301599274

"""




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
                    conditionalProb[(feat, label)] += k
                    counts[(feat, label)] += 2 * k

            # normalizing:
            prior.normalize()
            "**** YOUR CODE HERE to normalize conditionalProb "
            for key in conditionalProb:
                denomi = counts[key]
                if denomi > 0:
                    conditionalProb[key] = conditionalProb[key] / counts[key]
                else:
                    conditionalProb[key] = 0

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
            logJoint[label] = math.log(max(self.prior[label], 1e-10))
            for feat in datum.keys():
                prob = self.conditionalProb[(feat, label)]
                prob = max(prob, 1e-10)
                if datum[feat] > 0:
                    logJoint[label] += math.log(prob)
                else:
                    logJoint[label] += math.log(1 - prob)
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
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        return -1 if nn.as_scalar(score) < 0 else 1
    

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        condition = False
        while not condition:
            condition = True
            for (x, y) in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    condition = False



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
        self.hidden_size1 = 300
        self.hidden_size2 = 200
        self.hidden_size3 = 100

        self.learning_rate = 0.04
        self.batch_size = 10  

        self.W1 = nn.Parameter(1, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)

        self.W2 = nn.Parameter(self.hidden_size1, self.hidden_size2)
        self.b2 = nn.Parameter(1, self.hidden_size2)
        

        self.W3 = nn.Parameter(self.hidden_size2, self.hidden_size3)
        self.b3 = nn.Parameter(1, self.hidden_size3)
        

        self.W4 = nn.Parameter(self.hidden_size3, 1)
        self.b4 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1 = nn.Linear(x, self.W1)
        layer1_bias = nn.AddBias(layer1, self.b1)
        layer1_relu = nn.ReLU(layer1_bias)
        

        layer2 = nn.Linear(layer1_relu, self.W2)
        layer2_bias = nn.AddBias(layer2, self.b2)
        layer2_relu = nn.ReLU(layer2_bias)
        

        layer3 = nn.Linear(layer2_relu, self.W3)
        layer3_bias = nn.AddBias(layer3, self.b3)
        layer3_relu = nn.ReLU(layer3_bias)
        

        layer4 = nn.Linear(layer3_relu, self.W4)
        output = nn.AddBias(layer4, self.b4)
        
        return output

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
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
            Trains the model.
        """
        "*** YOUR CODE HERE ***"

        max_iterations = 5000
        target_loss = 0.019 

        initial_lr = self.learning_rate
        min_lr = 0.0005
        lr_decay_factor = 0.7

        patience = 40
        min_improvement = 0.0001

        current_lr = initial_lr
        best_loss = float('inf')
        consecutive_no_improvement = 0

        best_params = None
        
        for i in range(max_iterations):

            for x, y in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, 
                                                self.W3, self.b3, self.W4, self.b4])

                self.W1.update(gradients[0], -current_lr)
                self.b1.update(gradients[1], -current_lr)
                self.W2.update(gradients[2], -current_lr)
                self.b2.update(gradients[3], -current_lr)
                self.W3.update(gradients[4], -current_lr)
                self.b3.update(gradients[5], -current_lr)
                self.W4.update(gradients[6], -current_lr)
                self.b4.update(gradients[7], -current_lr)

            current_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))

            if current_loss < best_loss - min_improvement:
                best_loss = current_loss
                consecutive_no_improvement = 0

                best_params = {
                    'W1': self.W1.data.copy(),
                    'b1': self.b1.data.copy(),
                    'W2': self.W2.data.copy(),
                    'b2': self.b2.data.copy(),
                    'W3': self.W3.data.copy(),
                    'b3': self.b3.data.copy(),
                    'W4': self.W4.data.copy(),
                    'b4': self.b4.data.copy()
                }
            else:
                consecutive_no_improvement += 1

            if current_loss < target_loss:
                return

            if consecutive_no_improvement >= patience:
                if current_lr > min_lr:

                    current_lr *= lr_decay_factor
                    consecutive_no_improvement = 0
                else:

                    self.W1.data = best_params['W1']
                    self.b1.data = best_params['b1']
                    self.W2.data = best_params['W2']
                    self.b2.data = best_params['b2']
                    self.W3.data = best_params['W3']
                    self.b3.data = best_params['b3']
                    self.W4.data = best_params['W4']
                    self.b4.data = best_params['b4']
                    
                    current_lr = initial_lr * 0.5  
                    consecutive_no_improvement = 0
                    

                    restored_loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
                    if restored_loss < target_loss:
                        return



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
        self.hidden_size1 = 200
        self.hidden_size2 = 100
        self.learning_rate = 0.5
        self.batch_size = 100
        
        self.W1 = nn.Parameter(784, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)

        self.W2 = nn.Parameter(self.hidden_size1, self.hidden_size2)
        self.b2 = nn.Parameter(1, self.hidden_size2)

        self.W3 = nn.Parameter(self.hidden_size2, 10)
        self.b3 = nn.Parameter(1, 10)

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
        layer1 = nn.Linear(x, self.W1)
        layer1_bias = nn.AddBias(layer1, self.b1)
        layer1_relu = nn.ReLU(layer1_bias)
        
        layer2 = nn.Linear(layer1_relu, self.W2)
        layer2_bias = nn.AddBias(layer2, self.b2)
        layer2_relu = nn.ReLU(layer2_bias)

        layer3 = nn.Linear(layer2_relu, self.W3)
        output = nn.AddBias(layer3, self.b3)
        
        return output

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
        predicted = self.run(x)
        return nn.SoftmaxLoss(predicted, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:

            for x, y in dataset.iterate_once(self.batch_size):

                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                

                self.W1.update(gradients[0], -self.learning_rate)
                self.b1.update(gradients[1], -self.learning_rate)
                self.W2.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
                self.W3.update(gradients[4], -self.learning_rate)
                self.b3.update(gradients[5], -self.learning_rate)

            accuracy = dataset.get_validation_accuracy()
            if accuracy >= 0.975:  
                break

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


        self.hidden_size = 400
        self.learning_rate = 0.02
        self.batch_size = 40
        
        self.hidden_size = 400
        
     
        self.learning_rate = 0.02

        self.batch_size = 40
   
        self.Wx = nn.Parameter(self.num_chars, self.hidden_size)
        
        self.Wh = nn.Parameter(self.hidden_size, self.hidden_size)
 
        self.b = nn.Parameter(1, self.hidden_size)
        
        self.W2 = nn.Parameter(self.hidden_size, self.hidden_size)
        self.b2 = nn.Parameter(1, self.hidden_size)

        self.W3 = nn.Parameter(self.hidden_size, self.hidden_size//2)
        self.b3 = nn.Parameter(1, self.hidden_size//2)

        self.W_output = nn.Parameter(self.hidden_size//2, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))

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

            x_contribution = nn.Linear(x, self.Wx)
            
            h_contribution = nn.Linear(h, self.Wh)
            
            h = nn.ReLU(nn.AddBias(nn.Add(x_contribution, h_contribution), self.b))
        
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h, self.W2), self.b2)) 
        h3 = nn.ReLU(nn.AddBias(nn.Linear(h2, self.W3), self.b3))

        output = nn.AddBias(nn.Linear(h3, self.W_output), self.b_output)
        return output
    


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
        predicted = self.run(xs)
        return nn.SoftmaxLoss(predicted, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        "*** hint: User get_validation_accuracy() to decide when to finish learning ***"

        target_accuracy = 0.85
        max_epochs = 30

        initial_lr = self.learning_rate
        min_lr = 0.001

        patience = 6
        epochs_without_improvement = 0

        best_accuracy = 0
        best_weights = None
        
        current_lr = initial_lr



        for epoch in range(max_epochs):

            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.Wx, self.Wh, self.b, self.W2, self.b2, self.W3, self.b3, self.W_output, self.b_output])
                
                self.Wx.update(gradients[0], -current_lr)
                self.Wh.update(gradients[1], -current_lr)
                self.b.update(gradients[2], -current_lr)
                self.W2.update(gradients[3], -current_lr)
                self.b2.update(gradients[4], -current_lr)
                self.W3.update(gradients[5], -current_lr)
                self.b3.update(gradients[6], -current_lr)
                self.W_output.update(gradients[7], -current_lr)
                self.b_output.update(gradients[8], -current_lr)
            

            accuracy = dataset.get_validation_accuracy()
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                epochs_without_improvement = 0
                

                best_weights = {
                    'Wx': self.Wx.data.copy(),
                    'Wh': self.Wh.data.copy(),
                    'b': self.b.data.copy(),
                    'W2': self.W2.data.copy(),
                    'b2': self.b2.data.copy(),
                    'W3': self.W3.data.copy(),
                    'b3': self.b3.data.copy(),
                    'W_output': self.W_output.data.copy(),
                    'b_output': self.b_output.data.copy()
                }
            else:
                epochs_without_improvement += 1


            if epochs_without_improvement >= patience:
                current_lr *= 0.6
                if current_lr < min_lr:
                    break
                epochs_without_improvement = 0

            if best_accuracy >= target_accuracy:
                break

        if best_weights is not None:
            self.Wx.data = best_weights['Wx']
            self.Wh.data = best_weights['Wh']
            self.b.data = best_weights['b']
            self.W2.data = best_weights['W2']
            self.b2.data = best_weights['b2']
            self.W3.data = best_weights['W3']
            self.b3.data = best_weights['b3']
            self.W_output.data = best_weights['W_output']
            self.b_output.data = best_weights['b_output']
