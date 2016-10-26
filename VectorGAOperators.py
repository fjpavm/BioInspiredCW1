import numpy
import random

def randomUniform(in_min, in_max):
    in_min = float(in_min)
    in_max = float(in_max)
    return in_min + (in_max-in_min)*random.random()

class VectorGAOperators(object):
    def __init__(self, in_dimensions, in_deviationBase = 0.1, in_deviationExponents = None, in_minValues = None, in_maxValues = None):
        self.m_dimensions = in_dimensions
        if in_deviationExponents == None:
            in_deviationExponents = numpy.array([1.0])
        in_deviationExponents = numpy.array(in_deviationExponents).astype(float)
        self.m_deviationChoices = in_deviationBase**in_deviationExponents
        # use default -5.0 for COCO platform
        if in_minValues == None:
            in_minValues = numpy.array([-5]*in_dimensions).astype(float)
        # use default 5.0 for COCO platform
        if in_maxValues == None:
            in_maxValues = numpy.array([5]*in_dimensions).astype(float)
        self.m_minValues = numpy.array(in_minValues).astype(float)
        self.m_maxValues = numpy.array(in_maxValues).astype(float)
        self.m_range = self.m_maxValues - self.m_maxValues

    def mutate(self, in_vector):
        in_vector = numpy.array(in_vector).astype(float)
        deviation = random.choice(self.m_deviationChoices)
        # apply a random perturbation drawn from a normal function (= Gaussian with mean 0 and deviation 1)
        return numpy.fmax(self.m_minValues, numpy.fmin(self.m_maxValues, (in_vector + deviation*numpy.random.standard_normal(self.m_dimensions))))

    def boxcrossover(self, in_vector1, in_vector2):
        resultVector = []
        for dim in range(self.m_dimensions):
            value = randomUniform(in_vector1[dim], in_vector2[dim])
            resultVector.append(value)
        return numpy.array(resultVector).astype(float)

    def createRandom(self):
        return self.boxcrossover(self.m_minValues, self.m_maxValues)

# For testing code during development
if __name__ == "__main__":
    vecGA = VectorGAOperators(5, in_deviationExponents = [0, 1, 2, 4, 8])
    rand1 = vecGA.createRandom()
    rand2 = vecGA.createRandom()
    print 'createRandom1: ' + str(rand1)
    print 'createRandom2: ' + str(rand2)
    print 'boxcrossover: ' + str(vecGA.boxcrossover(rand1, rand2))
    print 'mutate1: ' + str(vecGA.mutate(rand1))