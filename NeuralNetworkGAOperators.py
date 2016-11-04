import NeuralNetwork

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

    def createRandom(self):
        pass

# For testing code during algorithm development
if __name__ == "__main__":
    nn = NeuralNetwork.NeuralNetwork([3,2,1])
    nn.setWeight(1, 0, 0, 1.0)
    nn.setWeight(2, 0, 0, 1.0)
    print nn.asString()
    res = nn([1, 2, 3])
    print 'result for [1, 2, 3]: ' + str(res)
    print 'result for [0, 0, 0]: ' + str(nn([0, 0, 0]))
    print ''
    samples = [([1,2,3],1.0),([0,0,0],0)]
    fitness = NeuralNetworkError(samples)
    print 'avgQErr = ' + str(fitness.avgQuadraticError(nn))