import sys
import os
sys.path.append(os.path.normpath('./coco_python'))
import time
import numpy as np
import fgeneric
import bbobbenchmarks
import GeneticAlgorithm
import VectorGAOperators
import NeuralNetwork
import NeuralNetworkGAOperators

datapathbase = 'GANN'

# dimensions = (2, 3, 5, 10, 20, 40)
# dimensions = (5,)
dimensions = (2, 10, 40)
function_ids = bbobbenchmarks.nfreeIDs
instances = range(1, 16) # + range(41, 51)

def createPopulationSelection(in_parentSelect, in_parentSelectParam):
    if in_parentSelect == 'T':
        return GeneticAlgorithm.TournamentSelection(in_parentSelectParam)
    if in_parentSelect == 'R':
        return GeneticAlgorithm.RankSelection(in_parentSelectParam)

def tryWithParameters(in_popSize, in_childrenRate, in_maxGenerations = 5000, in_parentRate = 0.01, in_parentSelect = 'T', in_parentSelectParam = 2, in_crossoverRate = 0.0, in_keepBest = False, in_alien = False):
    t0 = time.time()
    np.random.seed(int(t0))
    opts = dict(algid='GeneticAlgorithm',
            comments='')
    datapath = datapathbase + '_P' + str(in_popSize) + '_C' + str(in_childrenRate) + '_P' + str(in_parentRate) + '_' + in_parentSelect + str(in_parentSelectParam)
    if in_crossoverRate > 0.0:
        datapath += '_Cross' + str(in_crossoverRate)
    if in_keepBest == True:
        datapath += '_Best'
    if in_alien == True:
        datapath += '_Alien'

    f = fgeneric.LoggingFunction(datapath, **opts)
    for dim in dimensions:  # small dimensions first, for CPU reasons
        for fun_id in function_ids:
            for iinstance in instances:
                f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

                # prepare and run GA with parameters
                vectorGAOperators = VectorGAOperators.VectorGAOperators(dim, in_deviationBase = 0.01, in_deviationExponents = [0, 0.5, 1, 2, 3])
                parentSelection = createPopulationSelection(in_parentSelect, in_parentSelectParam)
                def fitness(x):
                    return - f.evalfun(x)
                ga = GeneticAlgorithm.GeneticAlgorithm(fitness,
                                                       in_createRandomIndividual = vectorGAOperators.createRandom,
                                                       in_mutateIndividual = vectorGAOperators.mutate,
                                                       in_crossoverIndividuals = vectorGAOperators.boxcrossover,
                                                       in_parentSelection = parentSelection,
                                                       in_parentRate = in_parentRate,
                                                       in_childrenRate = in_childrenRate,
                                                       in_keepBest = in_keepBest,
                                                       in_introduceAlien = in_alien,
                                                       in_populationSize = in_popSize)

                result = ga.run(in_maxGenerations = in_maxGenerations, in_targetFitness = -f.ftarget, in_staleStop = 100)
                print 'result: ' + str(result) + ' target: ' + str(-f.ftarget)
                f.finalizerun()
            print '      date and time: %s' % (time.asctime())
        print '---- dimension %d-D done ----' % dim


def trainNeuralNetwork(in_dim, in_function, in_numTrainingSamples = 500, in_test = False, in_fileToWrite = None):
    vectorGAOperators = VectorGAOperators.VectorGAOperators(in_dim)
    trainingSamples = []
    numTrainSamples = 0
    while numTrainSamples < in_numTrainingSamples:
            numTrainSamples += 1
            inputs = vectorGAOperators.createRandom()
            output = in_function(inputs)
            trainingSamples.append((inputs,output))
    trainingFunctions = NeuralNetworkGAOperators.NeuralNetworkError(trainingSamples)
    neuralNetworkGAOperators = NeuralNetworkGAOperators.NeuralNetworkGAOperators(in_dim, in_maxHiddenLayers = 4)
    parentSelection = createPopulationSelection('T', 5)
    def fitness(x):
        return -trainingFunctions.avgQuadraticError(x)
    ga = GeneticAlgorithm.GeneticAlgorithm(in_fitnessFunction = fitness,
                                           in_createRandomIndividual = neuralNetworkGAOperators.createRandom,
                                           in_mutateIndividual = neuralNetworkGAOperators.mutate,
                                           in_crossoverIndividuals = None,
                                           in_parentSelection = parentSelection,
                                           in_parentRate = 0.05,
                                           in_childrenRate = 0.10,
                                           in_keepBest = False,
                                           in_introduceAlien = True,
                                           in_populationSize = 1000)

    result = ga.run(in_maxGenerations = 10000, in_targetFitness = -0.0001, in_staleStop = 100)

    f = None
    if in_fileToWrite != None:
        f = open(in_fileToWrite, 'w')
    if in_test:
        testingSamples = []
        numTestingSamples = 0
        while numTestingSamples < in_numTrainingSamples:
            numTestingSamples += 1
            inputs = vectorGAOperators.createRandom()
            output = in_function(inputs)
            testingSamples.append((inputs,output))
        testingFunctions = NeuralNetworkGAOperators.NeuralNetworkError(testingSamples)
        print 'test value: ' + str(testingFunctions.avgQuadraticError(result[0][0]))
        if f != None: f.write('test value: ' + str(testingFunctions.avgQuadraticError(result[0][0])) + '\n')
    print 'neural network result: ' + str(result) 
    if f != None: f.write('neural network result: ' + str(result) + '\n')
    print result[0][0].asString()
    if f != None: f.write(result[0][0].asString() + '\n')
    if f != None: f.close()
    return result[0][0]

# popSize = 100
# childRate = 0.5
# tournamentSize = 20
# crossRate = 0.15
# aproximate maxGen from max function calls, population size and child rate
# maxGen =  round(150000.0/(popSize*childRate))
# tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_maxGenerations = maxGen, in_parentRate = 0.05, in_parentSelectParam = tournamentSize)

opts = dict(algid='GANeuralNetworkTraining',
            comments='')
f = fgeneric.LoggingFunction('NNTest', **opts)
separFunc = 1
lconFun = 6
hconFun = 10
multiFunc = 15
multi2Fun = 20
functionIds = [multi2Fun, multiFunc, hconFun, lconFun, separFunc]

for dim in dimensions:
    for funId in functionIds:
        f.setfun(*bbobbenchmarks.instantiate(funId, iinstance=1))
        trainNeuralNetwork(in_dim = dim, in_function = f, in_numTrainingSamples = dim*100, in_test = True, in_fileToWrite = 'NN_D' + str(dim) + '_f'+str(funId)+'.txt' )
        f.finalizerun()