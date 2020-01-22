import MPSPyLib as mps
import numpy as np
import sys
import os.path

def main(PostProcess=False):

    # Build operators
    Operators = mps.BuildSpinOperators(spin=1.0)

    # Define Hamiltonian MPO
    H = mps.MPO(Operators)
    H.AddMPOTerm('bond', ['splus', 'sminus'], hparam='J_xy', weight=0.5)
    H.AddMPOTerm('bond', ['sz','sz'], hparam='J_z', weight=1.0)

    # Ground state observables
    myObservables = mps.Observables(Operators)
    # Site terms
    myObservables.AddObservable('site', 'sz', 'z')
    # correlation functions
    myObservables.AddObservable('corr', ['sz', 'sz'], 'zz')
    myObservables.AddObservable('corr', ['splus', 'sminus'], 'pm')
    # Get correlation functions out to a distance of 1000
    myObservables.SpecifyCorrelationRange(1000)

    # Convergence parameters
    myConv = mps.iMPSConvParam(max_bond_dimension=12, variance_tol=-1.0,
                               max_num_imps_iter=1000)
    mod_list = ['max_bond_dimension','max_num_imps_iter']
    myConv.AddModifiedConvergenceParameters(0, mod_list, [20, 500])
    myConv.AddModifiedConvergenceParameters(0, mod_list, [40, 250])

    # Long run time (Enable if you prefer)
    #myConv.AddModifiedConvergenceParameters(0, mod_list, [60, 250])
    #myConv.AddModifiedConvergenceParameters(0, mod_list, [80, 250])

    L = 2
    # Define statics
    parameters = [{ 
        # Directories
        'job_ID'                    : 'Spin1.0Heisenberg',
        'Write_Directory'           : 'TMP_08/', 
        'Output_Directory'          : 'OUTPUTS_08/', 
        # System size and Hamiltonian parameters
        'L'                         : L,
        'J_z'                       : 1.0, 
        'J_xy'                      : 1.0, 
        'simtype'                   : 'Infinite',
        # Convergence parameters
        'MPSObservables'            : myObservables,
        'MPSConvergenceParameters'  : myConv,
        'logfile'                   : True
    }]

    # Write Fortran-readable main files
    MainFiles = mps.WriteFiles(parameters, Operators, H,
                               PostProcess=PostProcess)
    # Run the simulations
    if(not PostProcess):
        if os.path.isfile('./Execute_MPSMain'):
            RunDir = './'
        else:
            RunDir = None
        mps.runMPS(MainFiles, RunDir=RunDir)
        return

    # Postprocessing and plotting
    # ---------------------------

    Outputs = mps.ReadStaticObservables(parameters)
    clfilename = parameters[0]['job_ID'] + 'correlationLength.dat'
    clfile = open(clfilename, 'w')

    for Output in Outputs:
        chi = Output['max_bond_dimension']
        state = Output['state']
        print('Chi', chi, 'state', state,
              'energy density', Output['energy_density'])

        if(state == 0):
            corrfilename = parameters[0]['job_ID'] + 'chi' + str(chi) \
                           + 'corr.dat'
            corrfile = open(corrfilename, 'w')

            for ii in range(0, myObservables.correlation_range):
                corrfile.write('%16i'%(ii) + '%30.15E'%(Output['zz'][ii])
                               + '%30.15E'%(Output['pm'][ii]) + '\n')

            corrfile.close()
            clfile.write('%16i'%(chi) + '%30.15E'%(Output['Correlation_length'])
                         + '\n')
            print(sum(Output['z']), Output['zz'][0:6])

    clfile.close()

    return


if(__name__ == '__main__'):
    # Check for command line arguments
    Post = False
    for arg in sys.argv[1:]:
        key, val = arg.split('=')
        if(key == '--PostProcess'): Post = (val == 'T') or (val == 'True')

    # Run main function
    main(PostProcess=Post)


