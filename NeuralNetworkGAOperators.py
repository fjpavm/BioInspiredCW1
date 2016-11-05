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
        if False and mutationTypeRand > 0.05:
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
        if neuralNet.m_layerSizes[layer] == 1 :
            self.addNode(layer, neuralNet)
            return
        if neuralNet.m_layerSizes[layer] >= self.m_maxLayerSize :
            self.removeNode(layer, neuralNet)
            return
        addRemove = random.choice([self.addNode, self.removeNode])
        addRemove(layer, neuralNet)

    def addNode(self, layer, neuralNet):
        prevLayerSize = neuralNet.m_layerSizes[layer-1]
        layerSize = neuralNet.m_layerSizes[layer]
        nextLayerSize = neuralNet.m_layerSizes[layer+1]
        # add random column of size prevLayerSize+1 to matrix from layer-1 to layer
        prevMatrix = numpy.matrix( numpy.insert(neuralNet.getWeightMatrix(layer-1), layerSize, numpy.random.standard_normal(prevLayerSize+1), axis = 1 ) )
        # add random line of size nextLayerSize to matrix from layer to layer+1
        nextMatrix = numpy.matrix( numpy.insert(neuralNet.getWeightMatrix(layer), layerSize, numpy.random.standard_normal(nextLayerSize), axis = 0 ) )
        neuralNet.m_layerSizes[layer] = layerSize+1
        neuralNet.setWeightMatrix(layer-1, prevMatrix)
        neuralNet.setWeightMatrix(layer, nextMatrix)
        neuralNet.m_hiddenFunctions[layer-1].append( random.choice( self.m_possibleFunctions ) )

    def removeNode(self, layer, neuralNet):
        prevLayerSize = neuralNet.m_layerSizes[layer-1]
        layerSize = neuralNet.m_layerSizes[layer]
        nextLayerSize = neuralNet.m_layerSizes[layer+1]
        node = random.randrange(0, layerSize)
        # remove column for node from matrix from layer-1 to layer
        prevMatrix = numpy.matrix( numpy.delete(neuralNet.getWeightMatrix(layer-1), node, axis = 1 ) )
        # remove line for node from matrix from layer to layer+1
        nextMatrix = numpy.matrix( numpy.delete(neuralNet.getWeightMatrix(layer), node, axis = 0 ) )
        neuralNet.m_layerSizes[layer] = layerSize-1
        neuralNet.setWeightMatrix(layer-1, prevMatrix)
        neuralNet.setWeightMatrix(layer, nextMatrix)
        neuralNet.m_hiddenFunctions[layer-1].pop(node)


# For testing code during algorithm development
if __name__ == "__main__":
    nn = NeuralNetwork.NeuralNetwork([2,2,1])
    nn.setWeight(1, 0, 0, 1.0)
    nn.setWeight(2, 0, 0, 1.0)

    nnGAop = NeuralNetworkGAOperators(in_numInputs = 2, in_maxHiddenLayers=5, in_maxLayerSize = 3)
    # nn = nnGAop.createRandom()

    count = 1000
    while count > 0 :
        count -= 1
        print nn.asString()
        nn = nnGAop.mutate(nn)
        print nn.asString()
        samples = [([1.3,2.7],1.0),([0.0,1.0],0),([10.0,1.0],0),([0.5,1.0],0),([1.0,1.0],0),([0.0,5.0],0),([0.0,2.0],0)]
        fitness = NeuralNetworkError(samples)
        print 'avgQErr = ' + str(fitness.avgQuadraticError(nn))


    # res = nn([1, 2, 3])
    # print 'result for [1, 2, 3]: ' + str(res)
    # print 'result for [0, 0, 0]: ' + str(nn([0, 0, 0]))
    print ''
    # samples = [([1,2,3],1.0),([0,0,0],0)]
    # fitness = NeuralNetworkError(samples)
    # print 'avgQErr = ' + str(fitness.avgQuadraticError(nn))