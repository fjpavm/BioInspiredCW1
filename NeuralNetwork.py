import numpy
import math

# sigmoid with p = 1 (scaling all weights by p would have same effect as having p multiply x)
def sigmoid(x):
    return 1/(1+math.exp(-x))

def linear(x):
    return x

class NeuralNetwork(object):

    def __init__(self, in_layerSizes, in_hiddenFunctions = None):
        self.m_outputLayerIndex = len(in_layerSizes)-1
        self.m_numInputs = in_layerSizes[0]
        self.m_numOutputs = in_layerSizes[self.m_outputLayerIndex]
        self.m_numHiddenLayers = len(in_layerSizes)-2
        self.m_layerSizes = in_layerSizes
        if(in_hiddenFunctions == None):
            in_hiddenFunctions = []
            for layerIndex in range(1, self.m_outputLayerIndex):
                nodeFunctions = [sigmoid]*self.m_layerSizes[layerIndex]
                in_hiddenFunctions.append(nodeFunctions)
        self.m_hiddenFunctions = in_hiddenFunctions
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
        print 'value:' + str(in_value) + ' ' + str(self.m_layerMatices[in_layerIndex-1][in_previousLayerNode,in_node])
        self.m_layerMatices[in_layerIndex-1][in_previousLayerNode,in_node] = float(in_value)
        print self.m_layerMatices[in_layerIndex-1][in_previousLayerNode,in_node]

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
    nn = NeuralNetwork([3,2,1])
    nn.setWeight(1, 0, 0, 1.0)
    nn.setWeight(2, 0, 0, 1.0)
    for index in range(0,2):
        print 'matrix from ' + str(index) + ' to ' + str(index+1) + ':\n' +  str(nn.getWeightMatrix(index))
    print nn.asString()
    res = nn([1, 2, 3])
    print 'result for [1, 2, 3]: ' + str(res)
