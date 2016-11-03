import numpy
import math

# sigmoid with p = 1
def sigmoid(x):
    return 1/(1+math.exp(-x))

class NeuralNetwork(object):

    def __init__(self, in_layerSizes, in_hiddenFunctions):
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
            self.m_layerMatices.append(numpy.zeros((self.m_layerSizes[layerIndex-1], self.m_layerSizes[layerIndex])))

    def setWeightMatrix(self, in_startLayerIndex, in_matrix):
        self.m_layerMatices[in_startLayerIndex] = in_matrix

    def getWeightMatrix(self, in_startLayerIndex):
        return self.m_layerMatices[in_startLayerIndex]

    # hidden layer index is 1 for first hidden layer
    def setFunctionForNodeOnHiddenLayer(self, in_hiddenLayerIndex, in_nodeIndex, in_function):
        self.m_hiddenFunctions[in_hiddenLayerIndex-1][in_nodeIndex] = in_function

    def getFunctionForNodeOnHiddenLayer(self, in_hiddenLayerIndex, in_nodeIndex):
        return self.m_hiddenFunctions[in_hiddenLayerIndex-1][in_nodeIndex]
