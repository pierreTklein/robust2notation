###############################################################################
# Artificial neural network - made from scratch using numpy (and love)
###############################################################################
import numpy as np

##################### Activation functions #####################
# Identity activation
def identity(xVec):
    return xVec
def d_identity(xVec):
    return np.ones(np.shape(xVec))

# Sigmoid activation function
def sigmoid(xVec):
    denom = 1.0 + np.exp(-xVec)
    return np.divide(1, denom)

# Derivative of a sigmoid function
def d_sigmoid(xVec):
    return sigmoid(xVec) * (1.0 - sigmoid(xVec))

# Relu activation function
def relu(xVec):
    return np.maximum(0, xVec)

# Derivative of the relu activation function
def d_relu(xVec):
    return 1.0 * (xVec>0)

# Softmax and derivative
def softmax(xVec):
    expVec = np.exp(xVec)
    # If vector
    if np.shape(xVec)[1] == 1:
        return np.divide(expVec, np.sum(expVec))
    # If matrix
    else:
        transposedSum = np.sum(expVec, axis=1)
        transposedDivisor = np.divide(expVec.transpose(), transposedSum)
        return transposedDivisor.transpose()

def d_softmax(xVec):
    # NOTE: have not tested if it will work for matrices
    sf_xVec = softmax(xVec)
    return np.multiply(sf_xVec, np.subtract(1.0, sf_xVec))


## Store the above in dictionaries ##
activationFuncs = {}
activationFuncs['identity'] = identity
activationFuncs['d_identity'] = d_identity
activationFuncs['sigmoid'] = sigmoid
activationFuncs['d_sigmoid'] = d_sigmoid
activationFuncs['relu'] = relu
activationFuncs['d_relu'] = d_relu
activationFuncs['softmax'] = softmax
activationFuncs['d_softmax'] = d_softmax


##################### Loss functions #####################
def quadError(yTrue, yPred):
    return (0.5) * np.square( np.subtract(yPred, yTrue) )
def d_quadError(yTrue, yPred):
    return np.subtract(yPred, yTrue)

lossFuncs = {}
lossFuncs['quadratic'] = quadError
lossFuncs['d_quadratic'] = d_quadError


##################### Models #####################

class sequential_model:
    # Initialization
    def __init__(self, alpha=0.01, loss='quadratic'):
        # Learning hyper-parameters
        self.alpha = alpha
        self.lossFunc = lossFuncs[loss]
        self.d_lossFunc = lossFuncs['d_'+loss]

        # Parameter storage
        self.weights = []
        self.layerActivFuncs = [] # activation functions for each layer
        self.layerDerivActivFuncs = [] # derivative of activation functions for each layer
        self.layer_Primal_Out = []
        self.layer_Deriv_Out = []

        # Training history storage
        self.history = {}
        self.history['loss'] = np.empty(0)
        self.history['acc'] = np.empty(0)
        self.history['val_loss'] = np.empty(0)
        self.history['val_acc'] = np.empty(0)


    ## Methods to construct the NN ##
    # Initializing the input layer
    def input_layer(self, inputDim):
        self.layer_Primal_Out.append( np.empty(((inputDim+1),1)) )
        self.layer_Deriv_Out.append( np.empty(((inputDim+1),1)) )
        self.layerActivFuncs.append( activationFuncs['identity'] )
        self.layerDerivActivFuncs.append( activationFuncs['d_identity'] )

    def add_hidden(self, n_nodes, activation='relu'):
        # Compute the previous layer's index (i.e. current last layer)
        prev_l_idx = len(self.layer_Primal_Out) - 1
        # Previous layer's dimension
        prev_dim = len(self.layer_Primal_Out[prev_l_idx])

        # Initialize the weight matrix between current and past layer
        cur_w = np.random.normal(loc=0.0, scale=1.0, size=(prev_dim,n_nodes)) * 0.01
        self.weights.append( cur_w )

        # Current layer output and derivattives
        self.layer_Primal_Out.append( np.empty(( (n_nodes+1),1 )) )
        self.layer_Deriv_Out.append( np.empty(( (n_nodes+1),1 )) )

        # Activation functions for current layer output
        self.layerActivFuncs.append( activationFuncs[activation] )
        self.layerDerivActivFuncs.append( activationFuncs['d_'+activation] )

    def output_layer(self, n_classes, activation='softmax'):
        # Compute the previous layer's index (i.e. current last layer)
        prev_l_idx = len(self.layer_Primal_Out) - 1
        # Previous layer's dimension
        prev_dim = len(self.layer_Primal_Out[prev_l_idx])

        # Initialize the weight matrix between current and past layer
        cur_w = np.random.normal(loc=0.0, scale=1.0, size=(prev_dim,n_classes)) * 0.01
        self.weights.append( cur_w )

        # Current layer output and derivattives
        self.layer_Primal_Out.append( np.empty(( n_classes , 1 )) )
        self.layer_Deriv_Out.append( np.empty(( n_classes , 1 )) )

        # Activation functions for current layer output
        self.layerActivFuncs.append( activationFuncs[activation] )
        self.layerDerivActivFuncs.append( activationFuncs['d_'+activation] )


    # Summarize #TODO: make the model summary cleaner
    def summarize(self):
        for i in range(len(self.weights)):
            print('Index %d:\t' % i, end='')
            print(np.shape(self.layer_Primal_Out[i]), end='')
            print(' --- ', end='')
            print(np.shape(self.weights[i]), end='')
            print(' --->')
        print("Output layer: \t", end='')
        print(np.shape(self.layer_Primal_Out[-1]))

    ## Training and predictions ##
    def train(self, train_X, train_y, epochs=10, validation=None, verbose=True, verboMod=1):
        # Initialize some variables
        N_examples = np.shape(train_X)[0]
        feat_dim = np.shape(train_X)[1]
        N_classes = np.shape(train_y)[1] # Assume one-hot

        # Store loss and accuracies
        trainingLoss = np.empty(epochs)
        validationLoss = np.empty(epochs)
        trainingAccuracy = np.empty(epochs)
        validationAccuracy = np.empty(epochs)

        ## Iterate through epochs ##
        for epoch_idx in range(0,epochs):

            if verbose and (epoch_idx % verboMod == 0):
                print("Epoch %d/%d\t| " % (epoch_idx+1, epochs), end='')

            # Iterate through each training example
            for ex_i in range(N_examples):
                # Shape feature vector and forward propogate
                cur_xVec = np.reshape(train_X[ex_i,:], (feat_dim, 1))
                self._forwardProp(cur_xVec)

                # Shape label vector and back propogate
                cur_yVec = np.reshape(train_y[ex_i], (N_classes, 1))
                curAvgLoss = self._backProp(cur_yVec)

            # Compute training loss and accuracy this epoch
            trainingLoss[epoch_idx] = self.getLoss(train_X, train_y)
            trainingAccuracy[epoch_idx] = self.getAccuracy(train_X, train_y)

            # Compute validation loss, if available
            if validation != None:
                valid_X, valid_y = validation
                validationLoss[epoch_idx] = self.getLoss(valid_X, valid_y)
                validationAccuracy[epoch_idx] = self.getAccuracy(valid_X, valid_y)

            # Output
            if verbose and (epoch_idx % verboMod == 0):
                print("loss: %f\t| " % trainingLoss[epoch_idx], end='')
                print("acc: %f\t| " % trainingAccuracy[epoch_idx], end='')
                if validation != None:
                    print("val_loss: %f\t| " % validationLoss[epoch_idx], end='')
                    print("val_acc: %f\t| " % validationAccuracy[epoch_idx], end='')

                print()


        # Packge the metrics
        self.history['loss'] = np.concatenate((self.history['loss'], trainingLoss))
        self.history['acc'] = np.concatenate((self.history['acc'], trainingAccuracy))

        if validation != None:
            self.history['val_loss'] = np.concatenate((self.history['val_loss'], validationLoss))
            self.history['val_acc'] = np.concatenate((self.history['val_acc'], validationAccuracy))

        return self.history

    ## Model prediction and evaluation functions ##
    # Predicts probability
    def predictProb(self, X):
        for l in range(0, len(self.weights)):
            # Activate current layer
            X_activated = self.layerActivFuncs[l](X)
            # Append bias
            X_withBias = np.hstack(( X_activated , np.ones((len(X_activated),1)) ))
            # Linear transform
            X = np.dot( X_withBias , self.weights[l] )

        # Activation of the last layer
        networkOut = self.layerActivFuncs[-1](X)
        return networkOut


    # The loss
    def getLoss(self, X, true_y):
        # Predict y
        pred_y = self.predictProb(X)
        # Get a matrix of element-wise losses
        lossMat = self.lossFunc(true_y, pred_y)
        # Average losses
        return np.average(lossMat)

    # The accuracy
    def getAccuracy(self, X, true_y):
        # Predict y and get most likely predicted index
        pred_y = self.predictProb(X)
        pred_indeces = np.argmax(pred_y, axis=1)
        # Get most likley true indeces
        true_indeces = np.argmax(true_y, axis=1)
        # Get number of correct predictions
        return np.sum( (pred_indeces == true_indeces) ) / len(true_y)


    ## Internal training functions ##
    # feedforward computation (for single example)
    def _forwardProp(self, curInput_x):
        # Saving the inputs (output of first / input layer neurons)
        self.layer_Primal_Out[0] = np.vstack((curInput_x,1.0))

        # Compute the feedforward weighted computation to the next layer
        linTransOut = np.dot( self.weights[0].transpose(), self.layer_Primal_Out[0] )

        # Iterate through each subsequent layer
        for l in range(1, len(self.layer_Primal_Out)-1):
            # Apply activation for each neuron receiving input from previous layer
            activatedNeurons_noBias = self.layerActivFuncs[l](linTransOut)

            # Compute the derivative of the activation for later use
            d_activation_noBias = self.layerDerivActivFuncs[l](linTransOut)
            self.layer_Deriv_Out[l] = d_activation_noBias #NOTE: no bias added

            # Add bias to feedforward values and store
            self.layer_Primal_Out[l] = np.vstack((activatedNeurons_noBias, 1.0))

            # Compute feedforward matrix product for the next layer
            linTransOut = np.dot( self.weights[l].transpose(), self.layer_Primal_Out[l] )

        # Compute the output for the last layer
        self.layer_Primal_Out[-1] = self.layerActivFuncs[-1](linTransOut)
        self.layer_Deriv_Out[-1] = self.layerDerivActivFuncs[-1](linTransOut)


    # Backprop (for single examle)
    def _backProp(self, curInput_y):
        # Compute the loss and the derivative of the error at the last layer
        loss_vec = self.lossFunc(curInput_y, self.layer_Primal_Out[-1])
        d_error_vec = self.d_lossFunc(curInput_y, self.layer_Primal_Out[-1])

        # Compute the backprop error at the final layer
        backPropErrorVec = np.multiply( self.layer_Deriv_Out[-1] , d_error_vec )

        # Iterate backward through the layers (from last layer index to second layer index, comprehensively)
        for l in range( (len(self.weights)-1), 0, -1):
            # Save the old weights for backpropogation to earlier layer
            old_w = self.weights[l]

            # Compute the gradient for currents weights (i.e. between layer l and l+1)
            # ... using the back prop error from the l+1 layer
            gradient_w = np.dot( self.layer_Primal_Out[l], backPropErrorVec.transpose() )

            # Weight update for current layer
            delta_w = np.multiply( (-1.0 * self.alpha), gradient_w )
            self.weights[l] = np.add( self.weights[l], delta_w )

            # Back propogate the backpropr error to the current layer, getting rid of bias term
            weightedCurErrorVec = np.dot( old_w, backPropErrorVec )[:-1]
            # Update the backprop error to be used for the previous layer
            backPropErrorVec = np.multiply( self.layer_Deriv_Out[l] , weightedCurErrorVec )

        ## Gradients and weights for the first layer (i.e. between 1st and 2nd layers) ##
        gradient_w = np.dot( self.layer_Primal_Out[0], backPropErrorVec.transpose() )
        # Weight update for current layer
        delta_w = np.multiply( (-1.0 * self.alpha), gradient_w )
        self.weights[0] = np.add( self.weights[0], delta_w )


        # Compute and return the loss at the beginning of this back prop
        return np.average(loss_vec)
