import sys 
import os
sys.path.append(os.path.normpath('./coco_python'))
import time
import numpy as np
import fgeneric
import bbobbenchmarks

datapathbase = 'GA_'

dimensions = (2, 3, 5, 10, 20, 40)
function_ids = bbobbenchmarks.nfreeIDs 
instances = range(1, 6) + range(41, 51) 

maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes 
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 10000      # SET to zero if algorithm is entirely deterministic 


def run_optimizer(fun, dim, maxfunevals, ftarget=-np.Inf):
    """start the optimizer, allowing for some preparation. 
    This implementation is an empty template to be filled 
    
    """
    # prepare
    x_start = 8. * np.random.rand(dim) - 4
    
    # call, REPLACE with optimizer to be tested
    PURE_RANDOM_SEARCH(fun, x_start, maxfunevals, ftarget)

def PURE_RANDOM_SEARCH(fun, x, maxfunevals, ftarget):
    """samples new points uniformly randomly in [-5,5]^dim and evaluates
    them on fun until maxfunevals or ftarget is reached, or until
    1e8 * dim function evaluations are conducted.

    """
    dim = len(x)
    maxfunevals = min(1e8 * dim, maxfunevals)
    popsize = min(maxfunevals, 200)
    fbest = np.inf
    
    for _ in range(0, int(np.ceil(maxfunevals / popsize))):
        xpop = 10. * np.random.rand(popsize, dim) - 5.
        fvalues = fun(xpop)
        idx = np.argsort(fvalues)
        if fbest > fvalues[idx[0]]:
            fbest = fvalues[idx[0]]
            xbest = xpop[idx[0]]
        if fbest < ftarget:  # task achieved 
            break

    return xbest

t0 = time.time()
np.random.seed(int(t0))

def tryWithParameters(in_popSize):
    opts = dict(algid='PureRandomSearch',
            comments='')
   # f = fgeneric.LoggingFunction(datapathbase+'Pop'+str(in_popSize), **opts)
    f = fgeneric.LoggingFunction('PureRandomSearch', **opts)
    for dim in dimensions:  # small dimensions first, for CPU reasons
        for fun_id in function_ids:
            for iinstance in instances:
                f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))
    
                # independent restarts until maxfunevals or ftarget is reached
                for restarts in xrange(maxrestarts + 1):
                    if restarts > 0:
                        f.restart('independent restart')  # additional info
                    run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                                  f.ftarget)
                    if (f.fbest < f.ftarget
                        or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                        break
    
                f.finalizerun()
    
                print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                      'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                      % (fun_id, dim, iinstance, f.evaluations, restarts,
                         f.fbest - f.ftarget, (time.time()-t0)/60./60.))
    
            print '      date and time: %s' % (time.asctime())
        print '---- dimension %d-D done ----' % dim
    

tryWithParameters(100)