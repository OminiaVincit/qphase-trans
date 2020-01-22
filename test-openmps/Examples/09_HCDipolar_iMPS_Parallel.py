import MPSPyLib as mps
import numpy as np
from scipy.special import sici
from math import pi
import os
import sys

def main(PostProcess=False, muvals=50, tvals=50):
    """
    Simulation of an infinite HC dipolar model executed with MPI as data
    parallel simulation.

    **Arguments**

    PostProcess : boolean, optional
        ``False`` means we are running a simulation and so write files and a
        pbs script. ``True`` means we are just postprocessing a simulation,
        and so only generate the internal data we need for extracting
        observables.
    """

    # Build operators
    Operators = mps.BuildBoseOperators(1)

    # Define Hamiltonian MPO
    H = mps.MPO(Operators, PostProcess=PostProcess)
    H.AddMPOTerm('site', 'nbtotal', hparam='mu', weight=-1.0)
    H.AddMPOTerm('bond', ['bdagger', 'b'], hparam='t', weight=-1.0)
    invrcube = lambda x : 1.0 / (x**3)
    H.AddMPOTerm('InfiniteFunction', ['nbtotal','nbtotal'], hparam='U',
                 weight=1.0, func=invrcube, L=1000, tol=1e-9)
    H.printMPO()

    # ground state observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'nbtotal', 'n')
    # correlation functions
    myObservables.AddObservable('corr', ['nbtotal', 'nbtotal'], 'nn')
    myObservables.AddObservable('corr', ['bdagger', 'b'], 'spdm')
    # Get correlation functions out to a distance of 1000
    myObservables.SpecifyCorrelationRange(1000)

    # Convergence setup
    bond_dimensions = [12, 16, 20, 32]
    myConv = mps.iMPSConvParam(max_bond_dimension=bond_dimensions[0],
                               variance_tol=-1.0, max_num_imps_iter=1000)
    for ii in range(1, len(bond_dimensions)):
        myConv.AddModifiedConvergenceParameters(0, ['max_bond_dimension',
                                                    'max_num_imps_iter'],
                                                [bond_dimensions[ii], 500])

    L = 2
    U = 1.0
    mumin = 0.5
    mumax = 1.8
    muiter = np.linspace(mumin, mumax, muvals)

    tmin = 0.01
    tmax = 0.4
    titer = np.linspace(tmin, tmax, tvals)

    parameters = []

    for ii in range(muiter.shape[0]):
        mu = muiter[ii]

        for jj in range(titer.shape[0]):
            tt = titer[jj]

            parameters.append({
                'simtype'                   : 'Infinite',
                # Directories
                'job_ID'                    : 'HCDipolar',
                'unique_ID'                 : '_mu%2.4f_t%2.4f'%(mu, tt),
                'Write_Directory'           : 'TMP_09/',
                'Output_Directory'          : 'OUTPUTS_09/',
                # System size and Hamiltonian parameters
                'L'                         : L,
                'U'                         : U,
                'mu' : mu,
                't' : tt,
                #Convergence parameters
                'MPSObservables'            : myObservables,
                'MPSConvergenceParameters'  : myConv,
                'logfile'                   : True
            })

    comp_info = {
        'queueing' : 'slurm',
        'time' : '12:00:00',
        'nodes' : ['128'],
        'mpi' : 'srun',
        'ThisFileName' : os.path.abspath( __file__ )
    }

    MainFiles = mps.WriteMPSParallelFiles(parameters,
                                          Operators, H, comp_info,
                                          PostProcess=PostProcess)

    ## For serial a run as comparison
    # MainFiles = mps.WriteFiles(parameters, Operators, H,
    #                           PostProcess=PostProcess)
    # if(not PostProcess):
    #     if os.path.isfile('./Execute_MPSMain'):
    #         RunDir = './'
    #     else:
    #         RunDir = None
    #     mps.runMPS(MainFiles, RunDir=RunDir)
    #     return

    if(PostProcess):
        # extract observables
        Outputs = mps.ReadStaticObservables(parameters)

        # heading for where postprocessing is to be written
        outputstub = 'HCDOut/' + parameters[0]['job_ID']
        if(not os.path.isdir('./HCDOut')): os.makedirs('./HCDOut')

        scalarfilenames = []
        scalarfile = []
        i = 0

        # create a scalar file and file name for each \chi
        for chi in bond_dimensions:
            scalarfilenames.append(outputstub + 'chi' + str(chi) + 'scalar.dat')
            scalarfile.append(open(scalarfilenames[i], 'w+'))
            i += 1

        for Output in Outputs:
            mu = Output['mu']
            t = Output['t']
            if((Output['mu'] == mu) and (Output['t'] == t)):
                # Simulation metadata
                state = Output['state']
                chi = Output['bond_dimension']
                error_code = Output['error_code']
                Number_of_Ground_States = Output['N_Ground_States']
                of = Output['Orthogonality_Fidelity']
                trunc = Output['Truncation_error']
                tolerance = Output['Fixed_Point_Tolerance']
                converged = Output['converged']
                energy_density = Output['energy_density']
                correlation_length = Output['Correlation_length']

                if(error_code != 0):
                    print('Warning! Parameters ' + str(mu) + ' ' + str(t) + ' '
                          + str(state) + ' ' + str(chi) + ' have error code '
                          + str(error_code))
                if(not converged):
                    print('Warning! Parameters ' + str(mu) + ' ' + str(t) + ' '
                          + str(state) + ' ' + str(chi) + ' did not converge ', of,
                          trunc, tolerance)

                # average density of unit cell
                number = Output['n']
                ucdensity = sum(number) / Output['L']
                print('mu t n', mu, t, number, ucdensity)

                # connected correlation function
                raw_nn = Output['nn']
                nn_connected = []

                for ii in range(len(raw_nn)):
                    nn_connected.append(raw_nn[ii] - 0.0 * ucdensity**2)

                # Single-particle density matrix
                spdm = Output['spdm']

                # Scalar quantities
                if(state == 0):
                    sfile = scalarfile[Output['convergence_parameter'] - 1]
                    sfile.write('%30.15E'%(mu) + '%30.15E'%(t)
                                + '%30.15E'%(ucdensity) + '%30.15E'%(energy_density)
                                + '%30.15E'%(correlation_length)
                                + '%30.15E'%(1.0 * Number_of_Ground_States)
                                + '%30.15E'%(of) + '%30.15E'%(trunc) + '\n')

                # correlation functions
                corrfilename = outputstub + 'mu' + str(mu) + 't' + str(t) \
                               + 'chi' + str(chi) + str(state) + 'corr.dat'
                corrfile = open(corrfilename, 'w')
                for ii in range(1, myObservables.correlation_range):
                    corrfile.write('%16i'%(ii) + '%30.15E'%(nn_connected[ii])
                                   + '%30.15E'%(spdm[ii]) + '\n')
                corrfile.close()

        for ii in range(len(scalarfile)):
            scalarfile[ii].close()


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    muvals = 50
    tvals = 50

    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')
        if(key == '--muvals'): muvals = int(val)
        if(key == '--tvals'): tvals = int(val)

    main(PostProcess=Post, muvals=muvals, tvals=tvals)
