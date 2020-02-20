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
    @staticmethod
    def sequential_importance_sampling(particles, **kwargs):
        """
        Particle sequential importance sampling (no resampling)

        This @staticmethod will take a list of Particles, increment their indices, and returns
        """
        [particle._update_index(particle.index, **kwargs) for particle in particles]
        return particles


    @staticmethod
    def multinomial(particles, shadow_work_resample = True, **kwargs):
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

        if shadow_work_resample:
            particle_shadow_works = np.array([particle.shadow_work for particle in particles])
            mean_shadow_work = -logsumexp(-particle_shadow_works) + np.log(len(particle_shadow_works))

        updated_particles = Resamplers._update_particles(particles = particles,
                                                         updated_work = mean_work,
                                                         resampled_indices = resampled_indices,
                                                         updated_shadow_work = mean_shadow_work,
                                                         **kwargs)
        resampled_particles = []
        return particles

    @staticmethod
    def _update_particles(particles, updated_work, updated_shadow_work, resampled_indices, **kwargs):
        """
        Particle index resampler.
        This @staticmethod will take a list of particles, resampled indices, and resample the particles
        according to resampled indices.

        args
            particles : list of apophis.particles.Particle objects
                particles to update
            updated_work : float
                resampled work
            updated_shadow_work : float
                resampled mean shadow work
            resampled_indices : list of int
                resampled indices corresponding to particles

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles
        """
        for current_particle_index, resampling_particle_index in enumerate(resampled_indices):
            particle_copy = copy.deepcopy(particles[resampling_particle_index])
            particles[current_particle_index].resample(resampling_particle = particle_copy,
                                                      work = updated_work,
                                                      shadow_work = updated_shadow_work,
                                                      **kwargs)
        resampled_particles = particles
        return resampled_particles


class Observables(object):
    """
    Library class that holds several observables to be computed from particles.

    Note : all observables contain arguments for a COMPLETE particle datum.
    observable(works_t_1, incremental_works_t, works_t, sampler_states_t_1, sampler_state_t)
    args:
        works_t_1 : np.array
            floats of the unnormalized works at t-1
        incremental_works_t : np.array
            floats of the incremental_works at time t
        works_t : np.array
            floats of the unnormalized works at t; identical to works_t_1 + works_t
        sampler_states_t_1 : list of sampler_states
            sampler_states at t-1
        sampler_state_t : list of sampler_states
            sampler_states at t


    """
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
        pass


class Trailblazers(object):
    """
    Class of staticmethods to conduct trailblazing on a the target invariant parameter sequence
    """
    @staticmethod
    def binary_search(self,
                      observable,
                      configure_observable,
                      floor_threshold,
                      particles,
                      threshold,
                      start_parameters,
                      end_parameters,
                      max_iterations=20,
                      initial_guess = None,
                      precision_threshold = 1e-6
                      **kwargs):
        """
        conduct a binary search on the target invariant parameters
        WARNING : at present, this algorithm is only applicable to 1-dimensional start/end parameters
                  and to SMC algorithms wherein eta_(n-1) \approx pi_(n-1) and the reverse kernel L_(n-1)
                  is defined such that L_(n-1)(x_n, x_(n-1)) = pi_n(x_(n-1)) * K_n(x_(n-1), x_n) / pi_n(x_n)
                  so that the incremental work w_(increment, n) = gamma_n(x_(n-1)) / gamma_(n-1)(x_(n-1))
        args
            observable : function
                function defining the observable to be computed from self.particles
            configure_observable : function
                function to amend the observable to check for a threshold
            floor_threshold : float
                the floor_threshold for binary search
            particles : list of apophis.Particle objects
                particles whose next invariant lambda will be queried
            start_parameters: float
                start value of binary search; this corresponds to the parameter that defines particle.reduced_potential
            end_parameters: float
                end value of binary search
            max_iterations: int, default 20
                maximum number of interations to conduct
            initial_guess: float, default None
                guess where the threshold is achieved
            precision_threshold: float, default None
                precision threshold below which, the max iteration will break

        returns
            updated_parameter : float
                parameter defining target invariant thermodynamic state at time t


        WARNING : at present, this algorithm is only applicable to 1-dimensional start/end parameters
                  and to SMC algorithms wherein eta_(n-1) \approx pi_(n-1) and the reverse kernel L_(n-1)
                  is defined such that L_(n-1)(x_n, x_(n-1)) = pi_n(x_(n-1)) * K_n(x_(n-1), x_n) / pi_n(x_n)
                  so that the incremental work w_(increment, n) = gamma_n(x_(n-1)) / gamma_(n-1)(x_(n-1))
        """
        _base_end_val = end_parameters
        right_bound = end_parameters
        left_bound = start_parameters
        previous_cumulative_works = np.array([particle.work for work in particles])
        sampler_states_t_1 = [particle.sampler_state for particle in particles]
        sampler_states_t = None
        #set the start

        if initial_guess is not None:
            midpoint = initial_guess
        else:
            midpoint = (left_bound + right_bound) * 0.5

        for iteration in range(max_iterations):
            if iteration != 0:
                midpoint = (left_bound + right_bound) * 0.5
            #set the new target invariant thermodynamic_state parameters
            [particle.set_parameters(midpoint) for particle in particles]

            #now compute the incremental works  for all particles
            incremental_works = np.array([particle.compute_incremental_work(update_particle = False) for particle in particles])
            _observable_value = observable(works_t_1 = previous_cumulative_works,
                                           incremental_works_t = incremental_works,
                                           works_t = previous_cumulative_works + incremental_works,
                                           sampler_states_t_1 = sampler_states_t_1,
                                           sampler_states_t = sampler_states_t)

            _configured_observable_value = configure_observable(_observable_value)


            if _configured_observable_value <= floor_threshold: #
                right_bound = midpoint
            else:
                left_bound = midpoint

            if precision_threshold is not None:
                if abs(right_bound - left_bound) <= precision_threshold:
                    midpoint = right_bound
                    break

        updated_parameter = midpoint
        return updated_parameter



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
