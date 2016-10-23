import random

# Helper Functions

def fitnessCompare(x, y):
    return cmp(x[1],y[1])

# Class Definitions 

class GeneticAlgorithm(object):
    def __init__(self, in_fitnessFunction, in_createRandomIndividual, in_mutateIndividual, in_crossoverIndividuals = None, in_parentSelection = None, in_generationalSelection = None, in_populationSize = 100, in_parentRate = 0.01, in_childrenRate = 0.5, in_crossoverRate = 0.0, in_keepBest = False, in_introduceAlien = False):
        self.m_fitnessFunction = in_fitnessFunction
        self.m_createRandomIndividual = in_createRandomIndividual
        self.m_mutateIndividual = in_mutateIndividual
        self.m_crossoverIndividuals = in_crossoverIndividuals
        if in_parentSelection == None:
            in_parentSelection = ProbabilisticFitnessSelection()
        self.m_parentSelection = in_parentSelection
        if in_generationalSelection == None:
            in_generationalSelection = ChildrenAndBestCurrent()
        self.m_generationalSelection = in_generationalSelection
        self.m_populationSize = in_populationSize
        self.m_parentRate = in_parentRate
        self.m_childrenRate = in_childrenRate
        self.m_crossoverRate = in_crossoverRate
        self.m_keepBest = in_keepBest
        self.m_introduceAlien = in_introduceAlien
        self.m_populationList = list()
        self.m_best = None
        self.m_currentGeneration = 0
        self.m_overalBest = None
        self.m_overalBestGeneration = 0

    def initializePopulation(self, in_newPopulationSize = None):
        if in_newPopulationSize != None:
            self.m_populationSize = in_newPopulationSize
        self.m_populationList = list()
        newSize = 0
        self.m_best = None
        self.m_currentGeneration = 0
        self.m_overalBest = None
        self.m_overalBestGeneration = 0
        while newSize < self.m_populationSize:
            newSize += 1
            individual = self.m_createRandomIndividual()
            fitness = self.m_fitnessFunction(individual)
            self.m_populationList.append((individual,fitness))
            if self.m_best == None or self.m_best[1] < fitness:
                self.m_best = (individual,fitness)
        self.m_overalBest = self.m_best
        return self.m_populationList

    def advanceOneGeneration(self):
        # select parents where number is parentRate * population
        parents = set()
        numParents = int(min(max(1, round(self.m_parentRate*self.m_populationSize)), self.m_populationSize))
        selectedParents = self.m_parentSelection(self.m_populationList, numParents)
        # apply genetic operators to create children
        tryCrossover = self.m_crossoverRate > 0 and self.m_crossoverIndividuals != None and numParents > 1
        childList = []
        numChildren = int(min(max(1, round(self.m_childrenRate*self.m_populationSize)), self.m_populationSize))
        randRange = list(range(numChildren))
        random.shuffle(randRange)
        parentRandomOrder = [i%numParents for i in randRange]
        childIndex = 0
        while childIndex < numChildren:
            parentIndex = parentRandomOrder[childIndex]
            parent = selectedParents[parentIndex]
            child = None
            if tryCrossover == True and random.random() < self.m_crossoverRate:
                otherParentIndex = parentIndex
                while otherParentIndex == parentIndex : otherParentIndex = random.randrange(0, numParents)
                otherParent = selectedParents[otherParentIndex]
                child = self.m_crossoverIndividuals(parent[0], otherParent[0])
            else:
                child = self.m_mutateIndividual(parent[0])
            fitness = self.m_fitnessFunction(child)
            childList.append((child, fitness))
            childIndex += 1
        # population update
        numberToSelect = self.m_populationSize
        newGenerationList = []
        if self.m_keepBest == True:
            numberToSelect -= 1
            newGenerationList.append(self.m_best)
            self.m_populationList.remove(self.m_best)
        if self.m_introduceAlien == True:
            numberToSelect -= 1
            alien = self.m_createRandomIndividual()
            fitness = fitness = self.m_fitnessFunction(alien)
            newGenerationList.append((alien, fitness))
        self.m_populationList = newGenerationList + self.m_generationalSelection(self.m_populationList, childList, numberToSelect)
        # recalculate best
        self.m_best = None
        self.m_currentGeneration += 1
        for individual in self.m_populationList:
            if self.m_best == None or self.m_best[1] < individual[1]:
                self.m_best = individual
                if self.m_best[1] > self.m_overalBest[1]:
                    self.m_overalBest = self.m_best
                    self.m_overalBestGeneration = self.m_currentGeneration
        return self.m_populationList

    def run(self, in_maxGenerations = None, in_targetFitness = None, in_staleStop = None, in_populationSize = None):
        self.initializePopulation(in_populationSize)
        # have stale stop condition equal to pop size if no stop condition is given
        if in_maxGenerations == None and in_targetFitness == None and in_staleStop == None:
            in_staleStop = self.m_populationSize
        while True:
            if in_maxGenerations != None and self.m_currentGeneration >= in_maxGenerations:
                return self.m_overalBest, self.m_overalBestGeneration, 'maxGen', self.m_currentGeneration
            if in_targetFitness != None and self.m_overalBest >= in_targetFitness:
                return self.m_overalBest, self.m_overalBestGeneration, 'target', self.m_currentGeneration
            if in_staleStop != None and self.m_currentGeneration - self.m_overalBestGeneration >= in_staleStop:
                return self.m_overalBest, self.m_overalBestGeneration, 'stale', self.m_currentGeneration
            self.advanceOneGeneration()
        


class SelectionFunction(object):
    # simply returns firsts in list
    def __call__(self, in_populationList, numToSelect):
        return in_populationList[0:numToSelect]

# Allows duplicates
class TournamentSelection(SelectionFunction):
    def __init__(self, in_tournamentSize = 2):
        self.m_tournamentSize = in_tournamentSize

    # Selects according to tournament results
    def __call__(self, in_populationList, numToSelect):
        # select one at a time by tournament
        selectedList = [None]*numToSelect
        while numToSelect > 0:
            numToSelect -= 1
            selectedList[numToSelect] =  self.tournament(in_populationList)
        return selectedList

    def tournament(self, in_populationList):
        tournamentSize = self.m_tournamentSize
        best = None
        while tournamentSize > 0:
            tournamentSize -= 1
            individual = random.choice(in_populationList)
            if best == None or individual[1] > best[1]:
                best = individual
        return best

# Allows duplicates
class RankSelection(SelectionFunction):
    def __init__(self, in_rankBias = 1):
        self.m_rankBias = in_rankBias

     # Selects according to tournament results
    def __call__(self, in_populationList, numToSelect):
        rankSum, rankList = self.createRankList(in_populationList)
        # select one at a time by rank selection
        selectedList = [None]*numToSelect
        while numToSelect > 0:
            numToSelect -= 1
            index, individual = self.rankSelection(rankList, rankSum) 
            selectedList[numToSelect] =  individual
        return selectedList

    def createRankList(self, in_populationList):
        # make copy of original list
        in_populationList = list(in_populationList)
        in_populationList.sort(fitnessCompare)
        index = 0
        popSize = len(in_populationList)
        rankList = [None]*popSize
        rankSum = 0
        while index < popSize:
            rankList[index] = (in_populationList[index], (index+1)**self.m_rankBias)
            rankSum += rankList[index][1]
            index += 1
        return rankSum, rankList
    
    def rankSelection(self, in_rankList, in_rankSum):
        # make choice according to p(individual) = biasedRank(Individual)/sumBiasedRanks
        randChoice = in_rankSum*random.random()
        sumForChoice = 0.0
        index = 0
        for individual, rank in in_rankList:
            sumForChoice += rank
            if randChoice < sumForChoice :
                return index, individual
            index += 1

# Avoids duplicates
class ProbabilisticFitnessSelection(SelectionFunction):
    def __init__(self, in_dropMinPercentage = 0.1):
        # if in_dropMinPercentage is 0.0 then worse will never be chosen unless it's also the best
        super(ProbabilisticFitnessSelection, self).__init__()
        self.m_dropMinPercentage = float(in_dropMinPercentage)

    # Selects according to distribution based on fitness function
    def __call__(self, in_populationList, numToSelect):
        # make copy of original list
        in_populationList = list(in_populationList)
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
        # select one at a time removing from original list
        selectedList = [None]*numToSelect
        while numToSelect > 0:
            index, individual = self.chooseOne(sumAdjustedFitness, in_populationList, adjustedMin)
            in_populationList.pop(index)
            sumAdjustedFitness -= (individual[1] - adjustedMin)
            numToSelect -= 1
            selectedList[numToSelect] = individual
        return selectedList

    def chooseOne(self, in_sumAdjustedFitness, in_populationList, in_adjustedMin):
        # make choice according to p(individual) = adjustedFitness(Individual)/sumAdjustedFitness
        randChoice = in_sumAdjustedFitness*random.random()
        sumForChoice = 0.0
        index = 0
        for individual in in_populationList:
            sumForChoice += (individual[1] - in_adjustedMin)
            if randChoice < sumForChoice :
                return index, individual
            index += 1


class GenerationalSelectionFunction(object):
    # Needs implementation in derived classes to return list of size in_numToSelect
    def __call__(self, in_currentGeneration, in_children, in_numToSelect):
        pass

class ChildrenAndBestCurrent(GenerationalSelectionFunction):
  
    def __call__(self, in_currentGeneration, in_children, in_numToSelect):
        numChildren = len(in_children)
        if numChildren > in_numToSelect:
            return in_children[0:in_numToSelect]
        # make copy of original list
        in_currentGeneration = list(in_currentGeneration) 
        # sort current generation by fitness
        in_currentGeneration.sort(fitnessCompare, reverse=True)
        return in_children + in_currentGeneration[0:in_numToSelect-numChildren]

# For testing code during algorithm development
if __name__ == "__main__":
    def fit(ind):
        return -float(ind-50)*float(ind-50)
    def creatRand():
        return random.randint(-1000,1000)
    def mutate(ind):
        return random.choice([ind-1,ind+1])
    geneticAlgorithm = GeneticAlgorithm(fit, creatRand, mutate, in_parentSelection = RankSelection())
    print 'result: ' + str(geneticAlgorithm.run(in_maxGenerations = 10000, in_populationSize = 1000, in_staleStop = 100))

   