import sys 
import os
sys.path.append(os.path.normpath('./coco_python'))
import time
import numpy as np
import fgeneric
import bbobbenchmarks
import GeneticAlgorithm
import VectorGAOperators

datapathbase = 'GA'

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

def tryWithParameters(in_popSize, in_childrenRate, in_parentRate = 0.01, in_parentSelect = 'T', in_parentSelectParam = 2, in_crossoverRate = 0.0, in_keepBest = False, in_alien = False):
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

                result = ga.run(in_maxGenerations = 5000, in_targetFitness = -f.ftarget, in_staleStop = 100)
                print 'result: ' + str(result) + ' target: ' + str(-f.ftarget)
                f.finalizerun()
            print '      date and time: %s' % (time.asctime())
        print '---- dimension %d-D done ----' % dim
    

# population size and children rate experiment
childrenRateList = [ 0.5] 
popSizeList = [ 100]
tournamentSize = 2
#for childRate in childrenRateList:
#    for popSize in popSizeList:
#        print 'Start Pop:' + str(popSize) + ' Child Rate:' + str(childRate) + ' Tounament Size:' + str(tournamentSize)
#        tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize)

popSize = 100
childRate = 0.5
#tournamentSizeList = [5, 10, 20]
#for tournamentSize in tournamentSizeList:
#        print 'Start Pop:' + str(popSize) + ' Child Rate:' + str(childRate) + ' Tounament Size:' + str(tournamentSize)
#        tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize)

#biasList = [0.5, 1, 2, 4]
#for bias in biasList:
#        print 'Start Pop:' + str(popSize) + ' Child Rate:' + str(childRate) + ' Rank Bias:' + str(bias)
#        tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelect = 'R', in_parentSelectParam = bias)

tournamentSize = 20
#tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize)

#tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize, in_alien = True)

#crossRateList = [0.05, 0.1, 0.25, 0.5, 0.75]
#for crossRate in crossRateList:
#    tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize, in_crossoverRate = crossRate, in_alien = True)

crossRate = 0.15
tryWithParameters(in_popSize = popSize, in_childrenRate = childRate, in_parentRate = 0.05, in_parentSelectParam = tournamentSize, in_crossoverRate = crossRate, in_alien = True)

