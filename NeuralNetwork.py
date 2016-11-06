import numpy

# sigmoid with p = 1 (scaling all weights by p would have same effect as having p multiply x)
def sigmoid(x):
    return .5 * (1.0 - numpy.tanh(.5 * x))

def linear(x):
    return x

def derivative(function, x):
    if function == sigmoid:
        return sigmoid(x)*(1.0 - sigmoid(x))
    if function == linear:
        return 1

class NeuralNetwork(object):

    def __init__(self, in_layerSizes, in_hiddenFunctions = None):
        self.m_outputLayerIndex = len(in_layerSizes)-1
        self.m_numInputs = in_layerSizes[0]
        self.m_numOutputs = in_layerSizes[self.m_outputLayerIndex]
        self.m_numHiddenLayers = len(in_layerSizes)-2
        self.m_layerSizes = list(in_layerSizes)
        if(in_hiddenFunctions == None):
            self.m_hiddenFunctions = []
            for layerIndex in range(1, self.m_outputLayerIndex):
                nodeFunctions = [sigmoid]*self.m_layerSizes[layerIndex]
                self.m_hiddenFunctions.append(nodeFunctions)
        else:
            self.m_hiddenFunctions = []
            for layerIndex in range(1, self.m_outputLayerIndex):
                nodeFunctions = list(in_hiddenFunctions[layerIndex-1])
                self.m_hiddenFunctions.append(nodeFunctions)
        self.m_layerMatices = []
        for layerIndex in range(1, self.m_outputLayerIndex+1):
            self.m_layerMatices.append(numpy.matrix(numpy.zeros((self.m_layerSizes[layerIndex-1]+1, self.m_layerSizes[layerIndex]))))

    def setWeightMatrix(self, in_startLayerIndex, in_matrix):
        self.m_layerMatices[in_startLayerIndex] = numpy.matrix(in_matrix)

    def getWeightMatrix(self, in_startLayerIndex):
        return self.m_layerMatices[in_startLayerIndex]

    # hidden layer index is 1 for first hidden layer
    def setFunctionForNodeOnHiddenLayer(self, in_hiddenLayerIndex, in_nodeIndex, in_function):
        self.m_hiddenFunctions[in_hiddenLayerIndex-1][in_nodeIndex] = in_function

    def getFunctionForNodeOnHiddenLayer(self, in_hiddenLayerIndex, in_nodeIndex):
        return self.m_hiddenFunctions[in_hiddenLayerIndex-1][in_nodeIndex]

    # index 1 means from input to first hidden layer (or output for Neural Nets without hidden layers)
    def setWeight(self, in_layerIndex, in_previousLayerNode, in_node, in_value):
        self.m_layerMatices[in_layerIndex-1][in_previousLayerNode,in_node] = float(in_value)

    def getWeight(self, in_layerIndex, in_previousLayerNode, in_node):
        return self.m_layerMatices[in_layerIndex-1][in_previousLayerNode, in_node]

    # returns a numpy array of size equal to output layer size
    def __call__(self, in_inputs):
        currentLayerOutput = numpy.array(in_inputs).astype(float)
        for layerIndex in range(0, self.m_outputLayerIndex):
            currentLayerOutput = numpy.append(currentLayerOutput, 1.0)
            currentLayerOutput = (currentLayerOutput*self.m_layerMatices[layerIndex]).A1
            if layerIndex+1 < self.m_outputLayerIndex:
                for nodeIndex in range(0,self.m_layerSizes[layerIndex+1]):
                    currentLayerOutput[nodeIndex] = self.m_hiddenFunctions[layerIndex][nodeIndex](currentLayerOutput[nodeIndex])
        return currentLayerOutput

    # applies stochastic backpropagation based on given samples
    def backpropagate(self, in_samples, in_step = 0.1):
        # initialize delta weights matrices to zero
        layerUpdateMatices = []
        for layerIndex in range(1, self.m_outputLayerIndex+1):
            layerUpdateMatices.append(numpy.matrix(numpy.zeros((self.m_layerSizes[layerIndex-1]+1, self.m_layerSizes[layerIndex]))))
        for sample in in_samples:
            # forward propagate
            currentLayerOutput = numpy.array(sample[0]).astype(float)
            layerInputs = []
            layerOutputDerivative = []
            for layerIndex in xrange(0, self.m_outputLayerIndex):
                currentLayerOutput = numpy.append(currentLayerOutput, 1.0)
                layerInputs.append(currentLayerOutput)
                currentLayerOutput = (currentLayerOutput*self.m_layerMatices[layerIndex]).A1
                if layerIndex+1 < self.m_outputLayerIndex:
                    outputDerivative = []
                    for nodeIndex in range(0,self.m_layerSizes[layerIndex+1]):
                        outputDerivative.append( derivative(self.m_hiddenFunctions[layerIndex][nodeIndex], currentLayerOutput[nodeIndex]) )
                        currentLayerOutput[nodeIndex] = self.m_hiddenFunctions[layerIndex][nodeIndex](currentLayerOutput[nodeIndex])
                    layerOutputDerivative.append(numpy.array(outputDerivative))
            # backpropagate error
            layerOutputErrors = []
            outputError = sample[1] - currentLayerOutput
            layerOutputErrors.insert(0, outputError)
            for reverseLayerIndex in xrange(0, self.m_outputLayerIndex-1):
                layerIndex = self.m_outputLayerIndex-reverseLayerIndex
                outputError = (outputError*self.m_layerMatices[layerIndex-1].transpose()).A1
                outputError = layerOutputDerivative[layerIndex-2]*outputError[:-1]
                layerOutputErrors.insert(0, outputError)
            # calculate delta weights matrices 
            for layerIndex in xrange(0, self.m_outputLayerIndex):
                layerUpdateMatix = in_step*numpy.matrix(layerInputs[layerIndex]).transpose()*numpy.matrix(layerOutputErrors[layerIndex])
                layerUpdateMatices[layerIndex] += layerUpdateMatix
        # apply delta weights
        for layerIndex in xrange(0, self.m_outputLayerIndex):
            self.m_layerMatices[layerIndex] += layerUpdateMatices[layerIndex]

    def clone(self):
        nn = NeuralNetwork(self.m_layerSizes, self.m_hiddenFunctions)
        for layerIndex in xrange(0, self.m_outputLayerIndex):
            nn.m_layerMatices[layerIndex] = numpy.matrix(self.m_layerMatices[layerIndex])
        return nn

    def asString(self):
        stringResult = 'layer sizes:' + str(self.m_layerSizes) + '\n'
        for layerIndex in range(0, self.m_outputLayerIndex):
            stringResult += '\nlayer ' + str(layerIndex) + ' to ' + str(layerIndex+1) +'\n'
            stringResult += 'matrix\n' + str(self.m_layerMatices[layerIndex]) + '\n'
            if layerIndex+1 < self.m_outputLayerIndex:
                stringResult += 'functions:' + str(self.m_hiddenFunctions[layerIndex]) + '\n'
        return stringResult

# For testing code during algorithm development
if __name__ == "__main__":
    nn = NeuralNetwork([3,1])
    nn.setWeight(1, 0, 0, 0.01)
    nn.setWeight(1, 1, 0, 0.01)
    nn.setWeight(1, 2, 0, 0.01)
    nn.setWeight(1, 3, 0, 0.01)
    print nn.asString()
    count = 0
    while count < 10:
        count +=1
        nn.backpropagate([([1, 1, 1],1), ([0, 0, 0],0)])
        # print nn.asString()
        print str(nn([1, 1, 1])) + ' ' +  str(nn([0, 0, 0]))
    nnClone = nn.clone()
    print nnClone.asString()
    # nn.setWeight(2, 0, 0, 2.0)
    nn.setFunctionForNodeOnHiddenLayer(1, 1, linear)
    print nnClone.asString()
