"""Main module."""
#####Imports#####
import numpy as np
import logging
import os



# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("apophis")
_logger.setLevel(logging.DEBUG)

class SequentialMonteCarlo(object):
    """
    Super;
    The generalized algorithm proceeds as follows:

        Step 1: initialization-
            set n = 0;
            for i = 1, ..., N_n draw X_0^(i) ~ eta_0;
            evaluate { w_0(X_0^(i)) } with gamma_0(X_1) / eta_0(X_1);
            Iterate steps 2,3
        Step 2: resample-
            if Obs({W_n^(i), X_n^(i)}) exceeds some threshold T, resample particles and set W_n^(i) = 1/N_n
        Step 3: sampling-
            set n = n + 1; if n = p + 1 (i.e. the total sequence), stop;
            for i = 1, ..., N_n draw X_n^(i) ~ K_n(X_(n - 1)^(i), .);
            evaluate {w_n(X_(n-1:n)^(i))} with equation --> w_n(x_(n-1), x_n) = gamma_n(x_n) * L_(n-1)(x_n, x_(n-1)) / [gamma_(n-1)(x_(n-1)) * K_n(x_(n-1), x_n)]
            normalize works: W_n^(i) = W_(n-1)^(i) * w_n(X_(n-1 : n)^(i)) / sum_(N_n)[W_(n-1)^(j) * w_n(X_(n-1 : n))^(j)]

    Example:
    >>> # Below, we initialize the SMC with a target_invariant_thermodynamic_state, it's apprpriate sequence of parameters, and a propagator.
    >>> # However, the proposal invariant parameter sequence is undefined, so it defaults to the target_invariant_parameter_sequence.
    >>> # Also, eta_0 is left undefined, so we cannot generate an initial sampler_state;
    >>> # instead, we must define a sampler_state and propagate it w.r.t. the propagator defined at the 0th proposal_invariant_parameter_sequence,
    >>> # which is (as mentioned above) defaulted to the 0th target_invariant_parameter_sequence
    >>> smc = SequentialMonteCarlo(num_particles = 10,
    ...                            target_invariant_thermodynamic_state = target_invariant_thermodynamic_state,
    ...                            target_invariant_parameter_sequence = np.array([0., 0.2, 0.4, 0.6, 0.8, 1.0]),
    ...                            propagator = propagator,
    ...                            proposal_invariant_parameter_sequence = None,
    ...                            seed_sampler_states = [sampler_state],
    ...                            eta_0 = None
    ...                            )

    >>> smc.execute() #this executes the full protocol

    """

    def __init__(self,
                 num_particles,
                 target_invariant_thermodynamic_state,
                 target_invariant_parameter_sequence,
                 propagator,
                 proposal_invariant_parameter_sequence = None,
                 seed_sampler_states = None,
                 eta_0 = None,
                 **kwargs):
        """
        Generalized SMC __init__ method.

        arguments
            num_particles : int
                number of particles at 0th iteration
            target_invariant_thermodynamic_state : apophis.thermostates.ThermodynamicState
                thermodynamic state defining the target invariant distributions
            target_invariant_parameter_sequence : np.ndarray [n,l] or np.ndarray [1,l]
                the paramter sequence defining pi_0:n
                n = number of sequential target invariants
                l = dimension of parameters to define the target invariant thermodynamic_states, pi_0:n
            propagator : apophis.propagators.Propagator
                the propagator containing a proposal_invariant_thermodynamic_state (apophis.thermostates.ThermodynamicState)
                and an apophis.sampler_states.SamplerState seed.
            proposal_invariant_parameter_sequence : np.ndarray [n,m] or np.ndarray [1,m], default None
                the parameter sequence defining rho_0:n;
                n = number of sequential forward kernels K
                m = dimension of parameters to define the proposal invariant thermostate,  rho_n
            seed_sampler_states : list of apophis.sampler_states.SamplerState objects, default None
                list of sampler states to give to each particle; if None, they will be generated internally
            eta_0 : apophis.thermostates.ThermodynamicState, default None
                thermodynamic_state from which a SamplerState is generated and applied to each particle.
                If None, then the 0th entry of the proposal_invariant_parameter_sequence will parametrize the propagator proposal_invariant_thermodynamic_state
                and generate sampler states accordingly

        attributes
            iteration : 0
                first iteration
            kwargs : **kwargs
                extra arguments

        """
        from apophis.particles import Particle
        self.iteration = 0 # 0-indexed
        self.invariant_sequence_parameters = invariant_sequence_parameters
        self.termination_parameters = termination_parameters

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)



class OpenMMLayer(SequentialMonteCarlo):
    """
    Layer to make OpenMM compatible with SequentialMonteCarlo
    """
    def __init__(self,
                 system,
                 protocol_handler,
                 invariant_sequence_parameters,
                 temperature = 300 * unit.kelvin,
                 trajectory_directory = 'test',
                 trajectory_prefix = 'out',
                 atom_selection = 'not water',
                 topology = None,
                 ncmc_save_interval = 1,
                 **kwargs):
        """
        Initialization

        args
            system : openmm.System
                system
            protocol_handler : generalized protocol manager
                system-compatible protocol manager
            invariant_sequence_parameters : list
                list of invariant-defining parameters
            termination_parameters : type(invariant_sequence_parameters[0])
                termination parameters
            temperature : float * unit.kelvin
                target temperature
            trajectory_directory : str, default 'test'
                Where to write out trajectories resulting from the calculation. If None, no writing is done.
            trajectory_prefix : str, default 'out'
                What prefix to use for this calculation's trajectory files. If None, no writing is done.
            atom_selection : str, default not water
                MDTraj selection syntax for which atomic coordinates to save in the trajectories. Default strips
                all water.
            ncmc_save_interval : int, default 1
                interval with which to write ncmc trajectory.  If None, trajectory will not be saved.
        """
        #imports
        from perses.annihilation.lambda_protocol import RelativeAlchemicalState
        import openmmtools.mcmc as mcmc
        import openmmtools.integrators as integrators
        import openmmtools.states as states
        from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState

        super(OpenMMLayer, self).__init__(invariant_sequence_parameters, termination_parameters, **kwargs)


        #instantiate trajectory filenames
        if self.trajectory_directory and self.trajectory_prefix and self.topology is not None:
            self.write_traj = True
            self.eq_trajectory_filename = os.path.join(os.getcwd(), self.trajectory_directory, f"{self.trajectory_prefix}.eq.h5")
            self.neq_trajectory_filename = os.path.join(os.getcwd(), self.trajectory_directory, f"{self.trajectory_prefix}.neq.")
            self.atom_selection_indices = self.topology.select(self.atom_selection)
        else:
            self.write_traj = False
            self.eq_trajectory_filename, self.neq_trajectory_filename, self.atom_selection_indices = None, None, None

        #instantiate thermodynamic state
        lambda_alchemical_state = RelativeAlchemicalState.from_system(system)
        lambda_alchemical_state.set_alchemical_parameters(self.invariant_sequence_parameters[self.iteration], self.protocol_handler)
        self.target_invariant = CompoundThermodynamicState(ThermodynamicState(system, temperature = self.temperature), composable_states = [lambda_alchemical_state])
        self.proposal_invariant = copy.copy(self.target_invariant)



    def update_invariant(self, invariant, parameters, **kwargs):
        """
        Define an updated invariant with some lambda parameters

        args
            parameters : float or np array
                lambda parameters defining thermodynamics state
            invariant : generalized invariant object

        returns
            updated_invariant : generalized invariant object
                updated
        """
        updated_invariant.set_alchemical_parameters(parameters, lambda_protocol = self.protocol_handler)
        return updated_invariant


class AnnealedImportanceSampling(SequentialMonteCarlo):
    """
    Algorithm 1, developed by R. M. Neal as reported in https://arxiv.org/abs/physics/9803008 (a variant of Sequential Importance Sampling)
    """
    def __init__(self, **kwargs):
        super(AnnealedImportanceSampling, self).__init__(**kwargs)

    def compute_incremental_work(self, target_t, target_t_1, sampler_state_t_1, sampler_state_t = None, kernel_work = None, **kwargs):
        """
        Compute the incremental importance work for annealed importance sampling;
        Incremental weight is defined as w_t = target_t(sampler_state_t_1) / target_t_1(sampler_state_t_1).
        I.e. work_t = -ln(w_t).

        Here, we are allowed to propagate at target_t AFTER computing w_t and resampling particles

        args
            target_t : invariant
                invariant at time t
            target_t_1 : invariant
                invariant at time t - 1
            sampler_state_t_1 : sampler_state
                sampler state at t - 1
            sampler_state_t : sampler_state
                sampler state at t
            kernel_ratio : None
                L_(n-1) / K_n ratio

        returns
            incremental_weight : float
                -ln(w_t)
        """
        incremental_weight = target_t.reduced_potential(sampler_state_t_1) - target_t_1.reduced_potential(sampler_state_t_1)
        return incremental_weight

    def execute(self, num_initial_particles = 10, sample_gamma_1 = True, sampler_state_0 = None, **kwargs):
        """
        Algorithm executor.
        """
        while True:
            target_t = self.target_invariant
            if self.iteration == 0: #the only exception occurs in the initialization
                #first, we have to define a target invariant defined at iteration 1
                self.particles = list()
                if sample_gamma_1:
                    initial_target = target_t
                else:
                    initial_target = self.eta_1

                #then define the propagator whose invariant is eta_1 and propagate until num_particles i.i.d. samples are rendered
                for particle_index in range(len(num_initial_particles)):
                    if sample_gamma_1:
                        self.particles.append(Particle(index = particle_index, sampler_state = copy.deepcopy(sampler_state_0), work = 0.))
                    else:
                        sampler_state = self.generate_iid_sample(initial_target)
                        self.particles.append(Particle(index = particle_index), sampler_state = sampler_state)
                        self.particles[particle_index].update_work(target_t.reduced_potential(sampler_state) - initial_target.reduced_potential(sampler_state))
                continue

            #compute an incremental work
            [particle.append_work(incremental_work = self.compute_incremental_work(target_t, target_t_1, self.sampler_state))]

            #resample
            num_resamples = self.determine_num_resamples()
            self.resample(scheme = resample_scheme,
                          observable = resample_observable,
                          num_resamples = num_resamples,
                          threshold = resample_observable_threshold)

            #now we can propagate particles
            #first update the target invariant (and hence, the propagator invariant since it is defined by the target)
            self.determine_next_parameters()







            #then attempt to resample
            num_resamples = self.determine_num_resamples(**kwargs)
            self.resample(num_resamples)

            #then propagate
            propagator = self.generate_propagator()
            kernel_densities = self.propagate_particles(propagator, **kwargs)

            #then compute weights
            self.update_incremental_works(kernel_densities, target_t, target_t_1)

            target_t_1 = copy.copy(target_t) #previous target
            self.iteration += 1
            terminate = self.inquire_termination()
            if terminate:
                break
