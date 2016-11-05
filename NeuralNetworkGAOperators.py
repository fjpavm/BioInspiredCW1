import NeuralNetwork
import random
import numpy

class NeuralNetworkError(object):
    def __init__(self, samples):
        self.m_samples = samples
        self.m_invNumberSamples = 1.0/len(samples)

    def avgQuadraticError(self, neuralNetwork):
        avgQErr = 0.0
        for sample in self.m_samples:
            err = sample[1] - neuralNetwork(sample[0])
            avgQErr += err*err
        avgQErr = avgQErr * self.m_invNumberSamples
        return float(avgQErr)

class NeuralNetworkGAOperators(object):
    def __init__(self, in_numInputs, in_maxHiddenLayers = 1, in_numOutputs = 1, in_maxLayerSize = None):
        if in_maxLayerSize == None:
            in_maxLayerSize = in_numInputs*2
        self.m_maxHiddenLayers = in_maxHiddenLayers
        self.m_numInputs = in_numInputs
        self.m_numOutputs = in_numOutputs
        self.m_maxLayerSize = in_maxLayerSize
        self.m_possibleFunctions = [NeuralNetwork.sigmoid, NeuralNetwork.linear]

    def createRandom(self):
        numHiddenLayers = random.randint(0, self.m_maxHiddenLayers)
        layerSizes = [self.m_numInputs]
        for hiddeenLayerIndex in xrange(0, numHiddenLayers):
            layerSizes.append( random.randint(1, self.m_maxLayerSize) )
        layerSizes.append(self.m_numOutputs)
        hiddenFunctions = []
        for layerIndex in xrange(1, numHiddenLayers+1):
            nodeFunctions = []
            for nodeIndex in xrange(0, layerSizes[layerIndex]):
                nodeFunctions.append( random.choice(self.m_possibleFunctions) )
            hiddenFunctions.append(nodeFunctions)
        neuralNet = NeuralNetwork.NeuralNetwork(layerSizes, hiddenFunctions)
        for layerIndex in xrange(0, numHiddenLayers+1):
            mat = neuralNet.getWeightMatrix(layerIndex)
            mat = numpy.matrix( 100.0*numpy.random.standard_normal( mat.shape ) )
            neuralNet.setWeightMatrix(layerIndex, mat)
        return neuralNet

    def mutate(self, neuralNet):
        nn = neuralNet.clone()
        mutationTypeRand = random.random()
        if mutationTypeRand > 0.10 or nn.m_outputLayerIndex == 1:
            print 'weights'
            layer = random.randrange(0, nn.m_outputLayerIndex)
            self.mutateWeights(layer, nn)
            return nn
        if mutationTypeRand > 0.05:
            print 'function'
            layer = random.randrange(0, nn.m_outputLayerIndex)
            self.mutateFunction(layer, nn)
            return nn
        if True:
            print 'node'
            layer = random.randrange(1, nn.m_outputLayerIndex)
            self.mutateNodes(layer, nn)
            return nn

    def mutateWeights(self, layer, neuralNet):
        deviation = random.choice( [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6] )
        mat = neuralNet.getWeightMatrix(layer)
        mat += numpy.matrix( deviation*numpy.random.standard_normal( mat.shape ) )
        neuralNet.setWeightMatrix(layer, mat)

    def mutateFunction(self, layer, neuralNet):
        node = random.randrange(0, neuralNet.m_layerSizes[layer])
        functions = list(self.m_possibleFunctions)
        functions.remove( neuralNet.getFunctionForNodeOnHiddenLayer(layer, node) )
        func = random.choice(functions)
        neuralNet.setFunctionForNodeOnHiddenLayer(layer, node, func)

    def mutateNodes(self, layer, neuralNet):
        pass

    def addNode(self, layer, neuralNet):
        pass

    def removeNode(self, layer, neuralNet):
        pass



# For testing code during algorithm development
if __name__ == "__main__":
    nn = NeuralNetwork.NeuralNetwork([3,2,1])
    nn.setWeight(1, 0, 0, 1.0)
    nn.setWeight(2, 0, 0, 1.0)

    nnGAop = NeuralNetworkGAOperators(in_numInputs = 3, in_maxHiddenLayers=3, in_maxLayerSize = 3)
    nn = nnGAop.createRandom()

    print nn.asString()
    nn = nnGAop.mutate(nn)
    print nn.asString()
    res = nn([1, 2, 3])
    print 'result for [1, 2, 3]: ' + str(res)
    print 'result for [0, 0, 0]: ' + str(nn([0, 0, 0]))
    print ''
    samples = [([1,2,3],1.0),([0,0,0],0)]
    fitness = NeuralNetworkError(samples)
    print 'avgQErr = ' + str(fitness.avgQuadraticError(nn))