import MPSPyLib as mps
import MPSPyLib.EDLib as ed
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sys
import os.path


def main():
    """
    Example for a variational steady state search comparing the MPDO
    result to ED. Excited states are enabled as well for the variational
    OQS, but do not return the real part of the gap of the Liouvillian
    as we are using Ldagger L.
    """

    # Build spin operators for spin-1/2 system
    Ops = mps.BuildSpinOperators(0.5)
    Ops['sigmaz'] = 2 * Ops['sz']
    Ops['sigmax'] = (Ops['splus'] + Ops['sminus'])

    # Define Hamiltonian of transverse Ising model
    Ham = mps.MPO(Ops)
    Ham.AddMPOTerm('bond', ['sigmaz', 'sigmaz'], hparam='J', weight=-1.0)
    Ham.AddMPOTerm('site', 'sigmax', hparam='g', weight=-1.0)
    Ham.AddMPOTerm('lind1', 'splus', hparam='gamma', weight=1.0)

    # Convergence parameters 
    myConv = mps.MPDOConvParam(max_bond_dimension=64)
    ExConv = mps.MPSConvParam(max_bond_dimension=64)

    # Observables
    myObs = mps.Observables(Ops)
    myObs.AddObservable('site', 'sigmax', 'x')
    myObs.AddObservable('site', 'sigmaz', 'z')
    myObs.AddObservable('corr', ['sigmaz', 'sigmaz'], 'zz')

    # Set parameters.
    J = 1.0
    g = 1.5
    gamma = 0.1
    ll = 5

    parameters = []
    parameters.append({
        'simtype'                   : 'Finite',
        # Directories
        'job_ID'                    : 'Ising_Statics',
        'unique_ID'                 : 'g_' + str(g),
        'Write_Directory'           : 'TMP_36/',
        'Output_Directory'          : 'OUTPUTS_36/',
        # System size and Hamiltonian parameters
        'L'                         : ll,
	'J'                         : J,
        'g'                         : g,
        'gamma'                     : gamma,
        # ObservablesConvergence parameters
        'verbose'                   : 1,
        'MPSObservables'            : myObs,
        'MPSConvergenceParameters'  : myConv,
        'n_excited_states'          : 1,
        'eMPSConvergenceParameters' : ExConv,
        'logfile'                   : True
    })

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(parameters, Ops, Ham,
                               PostProcess=False)

    # Run the simulations and quit if we are not just Post
    if(not PostProcess):
        if os.path.isfile('./Execute_MPSMain'):
            RunDir = './'
        else:
            RunDir = None
        mps.runMPS(MainFiles, RunDir=RunDir)
        return

    # Do some ED measures
    # -------------------

    # Get Hamiltonian for energy measure and Liouville operator
    Hmat = Ham.build_hamiltonian(parameters[0], None, False, 33)
    Lmat = Ham.build_liouville(parameters[0], None, False, 2025)

    tmp = np.dot(np.transpose(np.conj(Lmat)), Lmat)
    [vals, vecs] = la.eigh(tmp)
    rvals = np.real(vals)

    idx = np.argsort(rvals)

    print('Two smallest eigenvalues', vals[idx[0]], vals[idx[1]])

    rho = np.reshape(vecs[:, idx[0]], [2**ll, 2**ll])
    rho /= np.trace(rho)

    energy_ED = np.real(np.trace(np.dot(Hmat, rho)))

    rho = ed.DensityMatrix(rho, 2, ll, None)
    xlist = []
    zlist = []
    zzlist = []
    zz = np.kron(Ops['sigmaz'], Ops['sigmaz'])

    for ii in range(ll):
        rhoii = rho.getset_rho((ii,))
        obsii = np.real(np.trace(np.dot(rhoii.rho, Ops['sigmax'])))
        xlist.append(obsii)

        obsii = np.real(np.trace(np.dot(rhoii.rho, Ops['sigmaz'])))
        zlist.append(obsii)

        if(ii == ll - 1): continue
        rhoij = rho.getset_rho((ii, ii + 1))
        obsii = np.real(np.trace(np.dot(rhoij.rho, zz)))
        zzlist.append(obsii)


    # PostProcess
    # -----------

    energy_MPS = []
    Outputs = mps.ReadStaticObservables(parameters)

    print('Eigenvalue from MPDOs', Outputs[0]['LdL_eigenvalue'],
          Outputs[1]['LdL_eigenvalue'])

    for Output in Outputs[:1]:
        energy_MPS.append(Output['energy'])
        print('x', Output['x'])
        print('z', Output['z'])

    zz = np.diag(Outputs[0]['zz'][1:, :-1])

    print('zz nearest neighbor', zz)
        
    print('energy MPS', energy_MPS)
    print('energy ED', energy_ED)

    print('')

    print('Error energy', np.abs(energy_MPS - energy_ED))
    print('Error x', np.abs(np.array(xlist) - Outputs[0]['x']))
    print('Error z', np.abs(np.array(zlist) - Outputs[0]['z']))
    print('Error NN zz', np.abs(zz - zzlist))

    return



if(__name__ == '__main__'):
    main()
