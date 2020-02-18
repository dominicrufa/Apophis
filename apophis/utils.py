"""Utility Module"""

#####Imports#####
import numpy as np
from scipy.special import logsumexp
import copy
from apophis.particles import Particle



class Resamplers(object):
    """
    Library class that holds several resampling schema, mostly in the form of @staticmethods.
    """
    def __init__(self, **kwargs):
        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    @staticmethod
    def sequential_importance_sampling(particles):
        """
        Particle sequential importance sampling (no resampling)

        This @staticmethod will take a list of Particles, increment their indices, and
        """

    @staticmethod
    def multinomial(particles):
        """
        Particle multinomial resampler.
        This @staticmethod will take a list of Particles, resample them, and return a list of updated Particles

        args
            particles : list of apophis.particles.Particle objects
                the particles to be resampled
            num_resamples : int, default
                number of

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles
        """
        particle_works = np.array([particle.work for particle in particles])
        normalized_weights = normalized_weights(particle_works)
        resampled_indices = np.random.choice(len(particles), len(num_particles), p = normalized_weights, replace = True)
        mean_work = -logsumexp(-particle_works) + np.log(len(particle_works))

        updated_particles = Resamplers._update_particles(particles = particles,
                                                         updated_work = mean_work,
                                                         resampled_indices = resampled_indices)
        resampled_particles = []
        return particles

    @staticmethod
    def _update_particles(particles, updated_work, resampled_indices, **kwargs):
        """
        Particle index resampler.
        This @staticmethod will take a list of particles, resampled indices, and resample the particles
        according to resampled indices.

        args
            particles : list of apophis.particles.Particle objects
                particles to update
            updated_work : float
                resampled work
            resampled_indices : list of int
                resampled indices corresponding to particles

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles

        TODO : check this for consistency (copy vs deepcopy)
        """
        for current_particle_index, resampling_particle_index in enumerate(resampled_indices):
            particle_copy = copy.deepcopy(particles[resampling_particle_index])
            particles[current_particle_index].resample(resampling_particle = particle_copy,
                                                      work = updated_work,
                                                      **kwargs)
        resampled_particles = particles
        return resampled_particles


class Observables(object):
    """
    Library class that holds several observables to be computed from particles.

    Note : all observables contain arguments for a COMPLETE particle datum.
    observable(works_t_1, incremental_works_t, works_t, sampler_states_t_1, sampler_states_t)
    args:
        works_t_1 : np.array
            floats of the unnormalized works at t-1
        incremental_works_t : np.array
            floats of the incremental_works at time t
        works_t : np.array
            floats of the unnormalized works at t; identical to works_t_1 + works_t
        sampler_state_t_1 : list
            configuration at t-1
        sampler_states_t : list
            configuration at t

    """
    def __init__(self, **kwargs):
        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    @staticmethod
    def nESS(works_t, **kwargs):
        """
        compute a normalized effective sample size
        returns
            nESS : float
                normalized effective sample size as defined by 1 / sum_1^N_t()
        """
        normalized_works = normalized_weights(works_t)
        nESS = 1. / (np.sum(normalized_weights**2) * len(normalized_weights))
        return nESS

    def covariance_matrix(works_t_1, incremental_works_t, **kwargs):
        """
        compute the differential of thermodynamic_length
        g is a Fisher information matrix; we presume that the matrix is only diagonally defined (i.e. there is only 1 lambda parameter)
        g_ij = sum_(i=1)^(N_t)[W_(t - 1)^(i) * w_t^(i)**2] where W_(t - 1)^(i) is the normalized previous weight of particle i and
        w_t^(i) is the incremental weight of particle i at time t

        TODO : generalize the Fisher information matrix and compute thermodynamic length
        """





#################
### Utilities ###
#################
def unnormalized_weights(works):
    """
    simple utility function to compute particle weights from an array of works

    args
        works : np.array
            unnormalized -log weights

    returns
        unnormalized_weights : np.array
    """
    unnormalized_weights = np.exp(-1 * works)
    return unnormalized_weights

def normalized_weights(works):
    """
    simple utility function to normalize an array of works

    args
        works : np.array
            unnormalized -log weights

    returns
        normalized_weights : np.array
            normalized_weights = np.exp(-1 * works - logsumexp(-1 * works))
    """
    unnormalized_weights = unnormalized_weights(works)
    normalized_weights = unnormalized_weights / np.exp(logsumexp(-1 * works))
    return normalized_weight

def binary_search(observable,
                  invariant,
                  invariant_updater,
                  incremental_work_calculator,
                  particles,
                  threshold,
                  start_parameters,
                  end_parameters,
                  max_iterations=100,
                  initial_guess = None,
                  precision_threshold = 1e-6
                  **kwargs):
    """
    observable : function
        function defining the observable to be computed from self.particles
    invariant : generalized invariant object
        particles : list of apophis.Particle objects
    invariant_updater : method
        method that updates the invariant
    incremental_work_calculator : method
        method to compute increment work
    particles : list of apophis.Particle objects
    threshold : float
        ceiling threshold value for observable
    start_parameters: float
        start value of binary search
    end_parameters: float
        end value of binary search
    max_iterations: int, default 20
        maximum number of interations to conduct
    initial_guess: float, default None
        guess where the threshold is achieved
    precision_threshold: float, default None
        precision threshold below which, the max iteration will break
    """
    _base_end_val = end_parameters
    right_bound = end_parameters
    left_bound = start_parameters
    invariant = invariant_updater(invariant, start_parameters)

    if initial_guess is not None:
        midpoint = initial_guess
    else:
        midpoint = (left_bound + right_bound) * 0.5

    for iteration in range(max_iterations):
        if iteration != 0:
            midpoint = (left_bound + right_bound) * 0.5

        updated_invariant = copy.copy(invariant_updater(invariant, midpoint))

        _incremental_works = [incremental_work_calculator(updated_invariant, invariant, sampler_state_t_1 = particle.sampler_state) for particle in particles]
        (works_t_1, incremental_works_t, works_t, sampler_state_t_1
                sampler_states_t : list

                kernel_work : np.array
                    -ln(L_(n-1)/K_n), or negative log of backward auxiliary kernel to the forward kernel

        _observable = observable()

        _observable, _incremental_works = self.compute_lambda_increment(new_val = midpoint,
                                         sampler_states = sampler_states,
                                         observable = observable,
                                         current_rps = current_rps,
                                         cumulative_works = cumulative_works)
        if _observable <= observable_threshold:
            right_bound = midpoint
        else:
            left_bound = midpoint

        if precision_threshold is not None:
            if abs(right_bound - left_bound) <= precision_threshold:
                midpoint = right_bound
                _observable, _incremental_works = self.compute_lambda_increment(new_val = midpoint,
                                                 sampler_states = sampler_states,
                                                 observable = observable,
                                                 current_rps = current_rps,
                                                 cumulative_works = cumulative_works)
                break


    return midpoint, _observable, _incremental_works
