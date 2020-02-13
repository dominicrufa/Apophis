"""Resampler Module"""

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
    def multinomial(particles, num_resamples, num_initial_particles, **kwargs):
        """
        Particle multinomial resampler.
        This @staticmethod will take a list of Particles, resample them, and return a list of updated Particles

        args
            particles : list of apophis.particles.Particle objects

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles
        """
        particle_works = np.array([particle.work for particle in particles])
        mean_particle_weight = np.average(particle_weights)
        normalized_weights = normalize_works(particle_weights)
        resampled_indices = np.random.choice(len(particles), num_resamples, p = normalized_weights, replace = True)

        updated_particles = Resamplers._update_particles(particles = particles, resampled_indices = resampled_indices, **kwargs)

        #now just update the weights
        [particle.update_weight(weight = mean_particle_weight) for particle in particles]
        return particles

    @staticmethod
    def _update_particles(particles, resampled_indices, **kwargs):
        """
        Particle index resampler.
        This @staticmethod will take a list of particles, resampled indices

        args
            particles : list of apophis.particles.Particle objects
                particles to update
            resampled_indices : list of int
                resampled indices corresponding to particles

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles
        """
        copied_particles = copy.deepcopy(particles)
        if len(resampled_indices) > len(particles):
            #then we have to resample up to len(particles), then deepcopy
            for current_idx, resampled_index in enumerate(resampled_indices[:len(particles)]):
                particle_to_copy = copied_particles[resampled_index]
                particles[current_idx].update_index(particle_to_copy.index)
                particles[current_idx].propagator = copy.deepcopy(particle_to_copy.propagator)
                particles[current_idx].update_sampler_state(copy.deepcopy(particle_to_copy.sampler_state))
                #we will update the propagator invariant later

            for resampled_index in resampled_indices[len(particles):]:
                particle_to_copy = copy.deepcopy(copied_particles[resampled_index])
                particles.append(Particle(index = particle_to_copy.index, sampler_state = copy.deepcopy(particle_to_copy.sampler_state)))
                #we will update the propagator invariant later

        else:
            for current_idx, resampled_index in resampled_indices:
                particle_to_copy = copied_particles[resampled_index]
                particles[current_idx].update_index(particle_to_copy.index)
                particles[current_idx].update_sampler_state(copy.deepcopy(particle_to_copy.sampler_state))
                #we will update the propagator invariant later

        del copied_particles #'cuz who needs em, anyway?'
        return particles



#################
### Utilities ###
#################

def normalize_works(works):
    """
    simple utility function to normalize an array of works

    args
        works : np.array of unnormalized -log weights

    returns
        normalized_weights : np.array
    """
    normalized_weights = np.exp(-1 * vector - logsumexp(-1 * vector))
    return normalized_weight

def average_work(works, num_initial_particles, **kwargs):
    """
    simple utility function to compute the average work of a work array

    args
        works : np.array
            unnormalized -log weights
        num_initial_particles : int
            number of particles initialized

    returns
        average_work : float
    """
    average_work = -logsumexp(-works) + np.log(num_initial_particles)
    return average_work
