import random

class GeneticAlgorithm(object):
    def __init__(self, in_fitnessFunction, in_createRandomIndividual, in_mutateIndividual, in_crossoverIndividuals = None, in_parentSelection = None, in_generationalSelection = None, in_populationSize = 100, in_parentRate = 1.0, in_crossoverRate = 0.0, in_mutationRate = 0.9, in_keepBest = False, in_introduceAlien = False):
        self.m_fitnessFunction = in_fitnessFunction
        self.m_createRandomIndividual = in_createRandomIndividual
        self.m_mutateIndividual = in_mutateIndividual
        self.m_crossoverIndividuals = in_crossoverIndividuals
        if in_parentSelection == None:
            in_parentSelection = ProbabilisticFitnessSelection()
        self.m_parentSelection = in_parentSelection
        if in_generationalSelection == None:
            in_generationalSelection = ProbabilisticFitnessSelection()
        self.m_generationalSelection = in_generationalSelection
        self.m_populationSize = in_populationSize
        self.m_parentRate = in_parentRate
        self.m_crossoverRate = in_crossoverRate
        self.m_mutationRate = in_mutationRate
        self.m_keepBest = in_keepBest
        self.m_introduceAlien = in_introduceAlien
        self.m_populationList = list()
        self.m_best = None

    def initializePopulation(self, in_newPopulationSize = None):
        if in_newPopulationSize != None:
            self.m_populationSize = in_newPopulationSize
        self.m_populationList = list()
        newSize = 0
        self.m_best = None
        while newSize < self.m_populationSize:
            newSize += 1
            individual = self.m_createRandomIndividual()
            fitness = self.m_fitnessFunction(individual)
            self.m_populationList.append((individual,fitness))
            if self.m_best == None or self.m_best[1] < fitness:
                self.m_best = (individual,fitness)
        return self.m_populationList

    def advanceOneGeneration(self):
        # select parents where number is parentRate * population
        parents = set()
        numParents = min(max(1, round(self.m_parentRate*self.m_populationSize)), self.m_populationSize)
        availableParents = []
        for i in range(self.m_populationSize):
            availableParents.append(i,self.m_populationList[i][1])
        for parent in range(numParents):


class SelectionFunction(object):
    # simply returns first in list
    def __call__(self, in_populationList):
        return 0, in_populationList[0]

class ProbabilisticFitnessSelection(SelectionFunction):
    def __init__(self, in_dropMinPercentage = 0.1):
        # if in_dropMinPercentage is 0.0 then worse will never be chosen unless it's also the best
        super(ProbabilisticFitnessSelection, self).__init__()
        self.m_dropMinPercentage = float(in_dropMinPercentage)

    # Selects according to distribution based on fitness function
    def __call__(self, in_populationList):
        # calculate min and max fitness
        minFitness = maxFitness = in_populationList[0][1]
        for individual in in_populationList:
            minFitness = individual[1] if individual[1] < minFitness else minFitness
            maxFitness = individual[1] if individual[1] > maxFitness else maxFitness
        fitnessRange = maxFitness - minFitness
        # protect against individuals all having the same fitness
        if fitnessRange <= 0:
            fitnessRange = 1.0
            minFitness = maxFitness - fitnessRange
        adjustedMin = minFitness - self.m_dropMinPercentage*fitnessRange
        # sum adjusted fitness
        sumAdjustedFitness = 0.0
        for individual in in_populationList:
            sumAdjustedFitness += (individual[1] - adjustedMin)
        # make choice according to p(individual) = adjustedFitness(Individual)/sumAdjustedFitness
        randChoice = sumAdjustedFitness*random.random()
        sumForChoice = 0.0
        index = 0
        for individual in in_populationList:
            sumForChoice += (individual[1] - adjustedMin)
            if randChoice < sumForChoice :
                return index, individual
            index += 1

# For testing code during algorithm development
if __name__ == "__main__":
    def fit(ind):
        return 50 - abs(float(ind)-50)
    def creatRand():
        return random.randint(1,100)
    def mutate(ind):
        return random.choice(ind-1,ind+1)
    geneticAlgorithm = GeneticAlgorithm(fit, creatRand, mutate)
    testPopList = geneticAlgorithm.initializePopulation()
    print 'Initial Population: '+ str(testPopList)
    print 'best: ' + str(geneticAlgorithm.m_best)
    simpleSelection = SelectionFunction()
    probabilisticSelection = ProbabilisticFitnessSelection()
    print 'simple selection: ' + str(simpleSelection(testPopList))
    for i in range(10):
        print 'probabilistic fitness selection ' + str(i) + ': ' + str(probabilisticSelection(testPopList))