"""Resampler Module"""

#####Imports#####
import numpy as np
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
    def multinomial(particles, num_particles, **kwargs):
        """
        Particle multinomial resampler.
        This @staticmethod will take a list of Particles, resample them, and return a list of updated Particles

        args
            particles : list of apophis.particles.Particle objects

        returns
            resampled_particles : list of apophis.particles.Particle objects
                resampled particles
        """
        particle_weights = np.array([particle.weight for particle in particles])
        mean_particle_weight = np.average(particle_weights)
        normalized_weights = normalize(particle_weights)
        resampled_indices = np.random.choice(len(particles), num_particles, p = normalized_weights, replace = True)

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
                particles[current_idx].update_configuration(particle_to_copy.configuration)

            for resampled_index in resampled_indices[len(particles):]:
                particle_to_copy = copied_particles[resampled_index]
                particles.append(Particle(index = particle_to_copy.index, configuration = particle_to_copy.configuration))

        else:
            for current_idx, resampled_index in resampled_indices:
                particle_to_copy = copied_particles[resampled_index]
                particles[current_idx].update_index(particle_to_copy.index)
                particles[current_idx].update_configuration(particle_to_copy.configuration)

        del copied_particles
        return particles



#################
### Utilities ###
#################

def normalize(vector):
    """
    simple utility function to normalize a vector

    args
        vector : np.array

    returns
        normalized_vector : np.array
    """
    Z = np.sum(vector)
    normalized_vector = vector / Z
    return normalized_vector
