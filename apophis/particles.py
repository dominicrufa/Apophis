"""Particle Module"""
#####Imports#####
import numpy as np
import copy

class Particle(object):
    """
    Generalized Particle object

    The Particle Object holds particle-specific information like ancestry, sampler_state, work, etc.
    """
    def __init__(self, index = 0, sampler_state = None, work = 0, shadow_work = 0, propagator = None, **kwargs):
        """
        Dummy init method to be overwritten.

        args
            index : int
                index label of particle
            sampler_state : generalized sampler_state object
                sampler_state of parameters
            work : float
                unnormalized work of particle
            propagator : generalized propagator object
                propagates particle
        """
        self.index = index
        self.work = work
        self.sampler_state = sampler_state
        self.propagator = propagator
        self.shadow_work = shadow_work

        self.ancestry = []
        self.sampler_states = []
        self.works = [work]
        self.shadow_works = [0] #initialize at zero because the first timestep does not have

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def update_sampler_state(self, sampler_state, record_sampler_state_history = False, **kwargs):
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

    def update_propagator_invariant(self, invariant, **kwargs):
        """
        Wrapper method to update the invariant of the propagator
        """
        self.propagator.invariant = invariant


    def update_index(self, index, record_index = True):
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

    def append_work(self, incremental_work, record_work = True, **kwargs):
        """
        Generalized method to update the work of a particle

        args
            incremental_work : float
                incremental work
            record_work : bool, default True
                whether to record the work history
        """
        self.work += incremental_work
        if record_work:
            self.works.append(self.work)

    def update_work(self, work, amend_work_history = True, **kwargs):
        """
        Generalized method to amend the previous work of a particle (i.e. if we were to resample)
        """
        self.work = work
        if amend_work_history:
            self.works[-1] = work


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
            self.update_sampler_state(new_state, **kwargs)

    def update_propagator(self, propagator, reset_shadow_work = True):
        """
        Utility method to update the propagator and the sampler state of the propagator from self.sampler_state

        args
            propagator : generalized propagator object
                propagates particle

        WARNING : be sure to update the propagator AFTER updating the sampler state
        """

        self.propagator = copy.deepcopy(propagator)
        self.propagator.sampler_state = self.sampler_state
        self.propagator.thermodynamic_state = propagator.thermodynamic_state
        if reset_shadow_work:
            try:
                self.propagator.reset_shadow_work() #reset the shadow work of the propagator
            except Exception as e:
                pass


    def resample(self,
                 resampling_particle = None,
                 work = None,
                 record_sampler_state_history = False, 
                 reset_shadow_work = True,
                 amend_work_history = True,
                 **kwargs):
        """
        Utility method to update the particle with the information of the resampled_particle

        args
            resampling_particle : apophis.particles.Particle object
                particle template to be copied to self
            work : float
                updated work of the current self.particle
        """
        self.update_index(resampling_particle.index, record_index = True)
        self.update_sampler_state(copy.deepcopy(resampling_particle.sampler_state), record_sampler_state_history = record_sampler_state_history)
        self.update_propagator(resampling_particle.propagator, reset_shadow_work = reset_shadow_work)
        self.update_work(work = work, amend_work_history = True)




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
