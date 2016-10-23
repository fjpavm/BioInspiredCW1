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

dimensions = (2, 3, 5, 10, 20, 40)
function_ids = bbobbenchmarks.nfreeIDs 
instances = range(1, 6) + range(41, 51) 

def createPopulationSelection(in_popSelect, in_popSelectParam):
    if in_popSelect == 'T':
        return GeneticAlgorithm.TournamentSelection(in_popSelectParam)
    if in_popSelect == 'B':
        return GeneticAlgorithm.RankSelection(in_popSelectParam)

def tryWithParameters(in_popSize, in_childrenRate, in_popSelect = 'T', in_popSelectParam = 2, in_crossoverRate = 0.0, in_keepBest = False, in_alien = False):
    t0 = time.time()
    np.random.seed(int(t0))
    opts = dict(algid='GeneticAlgorithm',
            comments='')
    datapath = datapathbase + '_P' + str(in_popSize) + '_C' + str(in_childrenRate) + '_' + in_popSelect + str(in_popSelectParam)
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
                vectorGAOperators = VectorGAOperators.VectorGAOperators(dim)
                parentSelection = createPopulationSelection(in_popSelect, in_popSelectParam)
                def fitness(x):
                    return - f.evalfun(x)
                ga = GeneticAlgorithm.GeneticAlgorithm(fitness, 
                                                       in_createRandomIndividual = vectorGAOperators.createRandom,
                                                       in_mutateIndividual = vectorGAOperators.mutate,
                                                       in_crossoverIndividuals = vectorGAOperators.boxcrossover,
                                                       in_parentSelection = parentSelection,
                                                       in_childrenRate = in_childrenRate,
                                                       in_keepBest = in_keepBest,
                                                       in_introduceAlien = in_alien,
                                                       in_populationSize = in_popSize)

                ga.run(in_targetFitness = -f.ftarget, in_staleStop = 100)
                 
                f.finalizerun()
            print '      date and time: %s' % (time.asctime())
        print '---- dimension %d-D done ----' % dim
    

# population size and children rate experiment
childrenRateList = [0.01, 0.25, 0.5, 0.75, 0.99] 
popSizeList = [10, 100, 1000]
for childRate in childrenRateList:
    for popSize in popSizeList:
        print 'Start Pop:' + str(popSize) + ' Child Rate:' + str(childRate)
        tryWithParameters(in_popSize = popSize, in_childrenRate = childRate)