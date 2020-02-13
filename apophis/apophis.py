"""Main module."""
#####Imports#####
import numpy as np
import logging
import os



# Instantiate logger
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("sMC_utils")
_logger.setLevel(logging.DEBUG)

class SequentialMonteCarlo(object):
    """
    Dummy super used to organize the algorithms and outline all of the necessary functionality that must be readily exposed.
    The generalized algorithm proceeds as follows:

    Step 1: initialization-
        set t = 1;
        for i = 1, ..., N_n draw X_1^(i) ~ eta_1;
        evaluate { w_1(X_1^(i)) } with gamma_1(X_1) / eta_1(X_1);
        Iterate steps 2,3
    Step 2: resample-
        if O({W_n^(i), X_n^(i)}) exceeds some threshold T, resample particles and set W_n^(i) = 1/N_n
    Step 3: sampling-
        set n = n + 1; if n = p + 1 (i.e. the total sequence), stop;
        for i = 1, ..., N_n draw X_n^(i) ~ K_n(X_(n - 1)^(i), .);
        evaluate {w_n(X_(n - 1:n)^(i))} with equation --> w_n(x_(n - 1), x_n) = gamma_n(x_n) * L_(n - 1)(x_n, x_(n - 1)) / [gamma_(n - 1)(x_(n - 1)) * K_n(x_(n - 1), x_n)]
        normalize works: W_n^(i) = W_(n - 1)^(i) * w_n(X_(n - 1 : n)^(i)) / sum_(N_n)[W_(n - 1)^(j) * w_n(X_(n - 1 : n))^(j)]

    Note : There are many variations of this algorithm, so I will not explicitly hardcode a sequence of iterations of this algorithm, but instead provide methods with **kwargs to allow for interoperability and subclassing
    """
    def __init__(self, **kwargs):
        """
        Dummy __init__ method.

        attributes
            iteration : 1
                first iteration
            kwargs : **kwargs
                extra arguments

        """
        from apophis.particles import Particle
        self.iteration = 1 # 1-indexed

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def resample(self, scheme, observable, num_resamples, threshold, **kwargs):
        """
        Generalized method to resample the particles with a given scheme, an observable, a threshold, and a number of particles to resample.

        args
            scheme : function
                function defining the resampling strategy
                NOTE : the scheme is designed to take a list of Particles, resample them, and return a list of updated Particles
            observable : function
                function defining the observable to be computed from self.particles
            num_particles : int
                number of particles to resample
            threshold : float
                generalized threshold of the observable to trigger resampling

        attributes
            particles : list of apophis.Particle objects
        """
        if observable(self.particles) > threshold:
            resampled_particles = scheme(particles = particles, num_resamples = num_resamples, **kwargs)
            self.particles = resampled_particles
        else:
            pass

    def determine_num_resamples(**kwargs):
        """
        Given particle weights (and perhaps an observable), determine the number of particles to resample

        args

        returns
            num_resamples : int, default len(num_particles)
                the number of particles that will be resampled
        """
        return len(self.particles)

    def update_particle_works(self, **kwargs):
        """
        Generalized method to compute and update works for all of the particles
        """
        pass

    def propagate_particles(self, propagator, **kwargs):
        """
        Generalized method to update Particles with the forward propagator K_n(x_(n-1), x_n) and (possibly) compute the kernel density

        args
            propagator : generalized propagator object

        returns
            kernel_densities : np.array
                list of kernel densities associated with the particles
        """
        kernel_densities = []
        for particle in self.particles:
            updated_sampler_state, kernel_density = propagator.propagate(sampler_state = particle.sampler_state, **kwargs)
            kernel_densities.append(kernel_density)
            particle.update_sampler_state(sampler_state = updated_sampler_state, **kwargs)

        return np.array(kernel_densities)

    def update_propagator(self, particle, propagator_invariant = None, sampler, move_type, **kwargs):
        """
        Generate a propagator from a sampler object and a invariant_object
        """
        if particle.propagator is None:
            move = move_type(**kwargs)
            propagator = sampler(propagator_invariant, particle.sampler_state, move = move)
            particle.propagator = propagator

        else:
            # the propagator invariant points from the propagator invariant
            pass

    def update_invariant(self, invariant, parameters, **kwargs):
        """
        Define an updated invariant with some lambda parameters

        args
            parameters : float or np array
                lambda parameters defining thermodynamics state
            invariant : generalized invariant object

        returns
            invariant : generalized invariant object
                updated
        """
        invariant.update(parameters)
        return invariant

    def generate_backward_propagator(self, **kwargs):
        """
        the backward propagator defines an auxiliary invariant
        which is never explicitly computed
        """
        pass

    def generate_iid_sample(self, invariant):
        pass

    def execute(self, num_initial_particles, **kwargs):
        """
        Algorithm executor.
        """
        #then iterate
        while True:
            target_t = self.generate_target_invariant(**kwargs)
            if self.iteration == 1: #the only exception occurs in the initialization
                #first, we have to define a target invariant defined at iteration 1
                self.particles = list()
                if sample_gamma_1:
                    initial_target = target_t
                else:
                    initial_target = self.generate_invariant(**kwargs)

                #then define the propagator whose invariant is eta_1 and propagate until num_particles i.i.d. samples are rendered
                propagator = self.generate_propagator(sampler = sampler, invariant_object = initial_target, **kwargs)

                for particle_index in range(len(num_initial_particles)):
                    updated_sampler_state, kernel_density = propagator.propagate(**kwargs)
                    work = 0. if sample_gamma_1 else gamma_1.reduced_potential(update_sampler_state) - initial_target.reduced_potential(updated_sampler_state)
                    self.particles.append(Particle(index = particle_index, sampler_state = updated_sampler_state, work = work))

            else:
                pass

            #then attempt to resample
            num_resamples = self.determine_num_resamples(**kwargs)
            self.resample(num_resamples)

            #then propagate
            propagator = self.generate_propagator()
            kernel_densities = self.propagate_particles(propagator, **kwargs)

            #then compute weights
            self.update_incremental_works(kernel_densities, target_t, target_t_1)

            target_t_1 = target_t #previous target
            self.iteration += 1
            terminate = self.inquire_termination()
            if terminate:
                break

class OpenMMLayer(SequentialMonteCarlo):
    """
    Layer to make OpenMM compatible with SequentialMonteCarlo
    """
    def __init__(self,
                 system,
                 protocol_handler,
                 temperature = 300 * unit.kelvin,
                 trajectory_directory = 'test',
                 trajectory_prefix = 'out',
                 atom_selection = 'not water',
                 topology = None,
                 ncmc_save_interval = 1):
        """
        Initialization

        args
            system : openmm.System
                system
            protocol_handler : generalized protocol manager
                system-compatible protocol manager
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

        super(OpenMMLayer, self).__init__(**kwargs)
        local_args = {key: val for key, val in locals() if key != 'self'}

        #update other kwargs that have not been set
        self.__dict__.update(local_args)

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
        lambda_alchemical_state = RelativeAlchemicalState.from_system(self.factory.hybrid_system)
        lambda_alchemical_state.set_alchemical_parameters(0.0, self.protocol_handler)
        self.target_invariant = CompoundThermodynamicState(ThermodynamicState(self.factory.hybrid_system, temperature = self.temperature), composable_states = [lambda_alchemical_state])
        self.proposal_invariant = copy.deepcopy(self.target_invariant)


    def update_invariant(self, invariant, parameters, **kwargs):
        """
        Define an updated invariant with some lambda parameters

        args
            parameters : float or np array
                lambda parameters defining thermodynamics state
            invariant : generalized invariant object

        returns
            invariant : generalized invariant object
                updated
        """
        invariant.set_alchemical_parameters(parameters, lambda_protocol = self.protocol_handler)
        return invariant


class AnnealedImportanceSampling(SequentialMonteCarlo):
    """
    Algorithm 1, developed by R. M. Neal as reported in https://arxiv.org/abs/physics/9803008 (a variant of Sequential Importance Sampling)
    """
    def __init__(self, **kwargs):
        super(AnnealedImportanceSampling, self).__init__(**kwargs)

    def execute(self, num_initial_particles = 10, sample_gamma_1 = True, sampler_state_0 = None, **kwargs):
        """
        Algorithm executor.
        """
        while True:
            target_t = self.target_invariant
            if self.iteration == 1: #the only exception occurs in the initialization
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



            #then attempt to resample
            num_resamples = self.determine_num_resamples(**kwargs)
            self.resample(num_resamples)

            #then propagate
            propagator = self.generate_propagator()
            kernel_densities = self.propagate_particles(propagator, **kwargs)

            #then compute weights
            self.update_incremental_works(kernel_densities, target_t, target_t_1)

            target_t_1 = target_t #previous target
            self.iteration += 1
            terminate = self.inquire_termination()
            if terminate:
                break
