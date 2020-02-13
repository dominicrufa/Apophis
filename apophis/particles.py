"""Particle Module"""
#####Imports#####
import numpy as np

class Particle(object):
    """
    Generalized Particle object

    The Particle Object holds particle-specific information like ancestry, sampler_state, work, etc.
    """
    def __init__(self, index = 0, sampler_state = None, work = 0, propagator = None, **kwargs):
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

        self.ancestry = []
        self.sampler_states = []
        self.works = [work]

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def update_sampler_state(self, sampler_state, record_history = False, **kwargs):
        """
        Wrapper method to update sampler_state from propagator

        args
            sampler_state : generalized sampler_state object
                the sampler_state of the particle for which we will update
            record_history : bool, default True
                whether to add the sampler_state to the history
        """
        self.sampler_state = sampler_state

        if record_history:
            self.sampler_states.append(copy.deepcopy(self.sampler_state))

        if hasattr(self, 'propagator'):
            self.propagator.sampler_state = self.sampler_state


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

    def append_work(self, incremental_work, record_work = True):
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

    def update_work(self, work, amend_work_history = True):
        """
        Generalized method to amend the previous work of a particle (i.e. if we were to resample)
        """
        self.work = work
        if amend_work_history:
            self.works[-1] = work


    def propagate(self, num_iterations, update_sampler_state = True):
        """
        propagate particles with a given propagator
        """
        self.propagator.run(num_iterations)

        if update_sampler_state:
            self.update_sampler_state()

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
