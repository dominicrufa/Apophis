"""Particle Module"""
#####Imports#####
import numpy as np
import copy

class Particle(object):
    """
    Generalized Particle object

    The Particle Object holds particle-specific information like ancestry, sampler_state, work, etc.
    """
    def __init__(self,
                 index,
                 iteration,
                 target_invariant_thermodynamic_state,
                 sampler_state,
                 propagator,
                 proposal_invariant_parameter_sequence,
                 target_invariant_parameter_sequence,
                 eta_0 = None
                 **kwargs):
        """
        Initialize a particle.

        args
            index : int
                index label of particle
            iteration : int
                iteration of SMC particle
            initial_work : float
                the initial work of sampling the particle from eta_0
                NOTE : this is always defined as gamma_0(x_0) / eta_0(x_0);
                if the particle sampler state is sampled directly from gamma_0(x_0), then the initial work is identically 0.
            target_invariant_thermodynamic_state : generalized thermodynamic state
                thermodynamic_state that defines pi_0:n
            initial_shadow_work : float, default 0
                initial shadow work, defined as ln(K_k(x_(k-1), x_k) / L_(k-1)(x_k, x_k-1)), or the log ratio of the forward to (auxiliary) backward kernel densities
                NOTE: this argument is exposed for resampling purposes; if initializing from the first iteration, there is no propagation;
                thus, the initial shadow work should be 0, identically
            sampler_state : generalized sampler_state object
                sampler_state
            propagator : generalized propagator object
                propagates particle with a kernel K_n defined with an invariant pi_n;
                is equipped with a thermodynamic state that is defined by rho_n, the proposal_thermostate
            proposal_invariant_parameter_sequence : np.ndarray [n,m] or np.ndarray [1,m]
                the parameter sequence defining rho_0:n;
                n = number of sequential forward kernels K
                m = dimension of parameters to define the proposal invariant thermostate,  rho_n
            target_invariant_parameter_sequence : np.ndarray [n,l] or np.ndarray [1,l]
                the paramter sequence defining pi_0:n
                n = number of sequential target invariants
                l = dimension of parameters to define the target invariant thermodynamic_states, pi_0:n
            eta_0 : generalized thermodynamic state
                the zeroth importance weight
                (NOTE: eta_0:n ALWAYS appears in the importance weight; however, in the 0th iteration, there is no propagation, only an importance weight
                    from eta_0 to gamma_0)
        """
        #Define current iteration information
        self._update_index(index)
        self.iteration = iteration
        #set target invariant
        self.target_invariant_thermodynamic_state = target_invariant_thermodynamic_state
        self.target_invariant_thermodynamic_state.set_parameters(target_invariant_parameter_sequence[self.iteration])
        self.target_invariant_parameter_sequence = target_invariant_parameter_sequence
        #set sampler_state
        self.sampler_state = sampler_state
        #set propagator
        self._update_propagator(propagator)
        #set proposal_invariant
        self.proposal_invariant_thermodynamic_state = self.propagator.thermodynamic_state
        self.proposal_invariant_thermodynamic_state.set_parameters(proposal_invariant_parameter_sequence[self.iteration])
        self.proposal_invariant_parameter_sequence = proposal_invariant_parameter_sequence

        #Define containers...
        self.ancestry = [index]
        self.sampler_states = [copy.deepcopy(sampler_state)]
        self.works = []
        self.shadow_works = [0] #initialize at zero because the first timestep does not have a propagation
        #initial work:
        if self.iteration == 0:
            self.reduced_potential = self.target_invariant_thermodynamic_state.reduced_potential(self.sampler_state) #corresponding to gamma_(iteration)(x_iteration)
            if eta_0 is not None:
                initial_work = self.reduced_potential - eta_0.reduced_potential(self.sampler_state)
            else:
                #we will assume that the sampler stae
                initial_work = 0.

            self._update_work(initial_work)

        else:
            pass

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def _update_sampler_state(self, sampler_state, record_sampler_state_history = False):
        """
        Wrapper method to update sampler_state

        args
            sampler_state : generalized sampler_state object
                the sampler_state of the particle for which we will update
            record_sampler_state_history : bool, default False
                whether to add the deepcopy of the sampler_state to the sampler state history
        """
        self.sampler_state = sampler_state

        if record_history:
            self.sampler_states.append(copy.deepcopy(self.sampler_state))

    def _update_index(self, index, record_index = True):
        """
        Generalized method to update the ancestry of a particle

        args
            index : int
                new index of the particle
            record_index : bool, default True
                whether to add the index to the ancestry
        """
        self.index = index

        if record_index:
            self.ancestry.append(index)

    def _update_work(self, incremental_work, record_work = True):
        """
        Generalized method to update the work of a particle

        args
            incremental_work : float
                incremental work
            record_work : bool, default True
                whether to record the work history
        """
        if hasattr(self, 'work'):
            self.work += incremental_work
        else:
            self.work = incremental_work

        if record_work:
            self.works.append(self.work)


    def _rectify_works(self, work, shadow_work, amend_work_histories = True, **kwargs):
        """
        Generalized method to amend the previous work of a particle (i.e. if we were to resample)
        """
        self.work = work
        self.shadow_work = shadow_work
        if amend_work_history:
            self.works[-1] = work
            self.shadow_works[-1] = shadow_work


    def propagate(self, num_iterations, update_sampler_state = True, record_shadow_work = True, **kwargs):
        """
        propagate particles with a given propagator
        """
        self.propagator.run(num_iterations)
        if record_shadow_work:
            try:
                self.shadow_work += self.propagator.get_shadow_work(dimensionless = True)
                self.shadow_works.append(self.shadow_work)
            except Exception as e:
                pass

        if update_sampler_state:
            new_state = self.propagator.sampler_state
            self._update_sampler_state(new_state, **kwargs)

    def _update_propagator(self, propagator, reset_shadow_work = True):
        """
        Utility method to update the propagator and the sampler state of the propagator from self.sampler_state

        args
            propagator : generalized propagator object
                propagates particle

        WARNING : be sure to update the propagator AFTER updating the sampler state
        """

        self.propagator = copy.deepcopy(propagator)
        self.propagator.sampler_state = self.sampler_state
        self.propagator.thermodynamic_state = self.proposal_invariant_thermodynamic_state
        if reset_shadow_work:
            try:
                self.propagator.reset_shadow_work() #reset the shadow work of the propagator
            except Exception as e:
                pass


    def resample(self,
                 resampling_particle,
                 work,
                 shadow_work,
                 record_sampler_state_history = False,
                 reset_shadow_work = True,
                 amend_work_histories = True,
                 **kwargs):
        """
        Utility method to update the particle with the information of the resampled_particle

        args
            resampling_particle : apophis.particles.Particle object
                particle template to be copied to self
            work : float
                updated work of the current self.particle
        """
        self._update_index(resampling_particle.index, record_index = True)
        self._update_sampler_state(copy.deepcopy(resampling_particle.sampler_state), record_sampler_state_history = record_sampler_state_history)
        self._update_propagator(resampling_particle.propagator, reset_shadow_work = reset_shadow_work)
        self._rectify_work(work = work, shadow_work = shadow_work, amend_work_histories = amend_work_histories)
        #and the target_invariant_thermodynamic_state is preserved
        #the last thing we need to do is update the reduced potential
        self.reduced_potential = resampling_particle.reduced_potential

    def compute_incremental_work(self, update_particle = True, **kwargs):
        """
        compute the incremental work as defined by u_n(x_n) - u_(n-1)(x_(n-1)) + ln(K_n(x_(n-1), x_n) / L_(n-1)(x_n, x_n-1))
        args
            update_particle : bool, default True
                whether to update the particle with the computed incremental work internally;
                if True, this will call _update_work AND set the new reduced_potential
        returns
            incremental_work : float
                the incremental_work
        """
        new_reduced_potential = self.target_invariant_thermodynamic_state.reduced_potential(self.sampler_state)
        old_reduced_potential = self.reduced_potential
        potential_difference = new_reduced_potential - old_reduced_potential
        incremental_shadow_work = self.shadow_works[-1] - self.shadow_works[-2]
        incremental_work = potential_difference + incremental_shadow_work
        if update_particle:
            self._update_work(incremental_work)
            self.reduced_potential = new_reduced_potential #we have to update the reduced potential for the next iteration
        else:
            pass

        return incremental_work 


class OpenMMParticle(Particle):
    """
    OpenMM supported particle
    """
    def __init__(self, index = 0, sampler_state = None, work = 0, propagator = None, **kwargs):
        super(OpenMMParticle, self).__init__(index = index, sampler_state = sampler_state, work = work, propagator = propagator, **kwargs)

    def update_propagator_invariant(self, invariant, **kwargs):
        """
        Wrapper method to update the invariant of the propagator
        """
        self.propagator.thermodynamic_state = invariant
