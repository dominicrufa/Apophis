"""Main module."""
#####Imports#####
import numpy as np
import logging



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
        normalize weights: W_n^(i) = W_(n - 1)^(i) * w_n(X_(n - 1 : n)^(i)) / sum_(N_n)[W_(n - 1)^(j) * w_n(X_(n - 1 : n))^(j)]

    Note : There are many variations of this algorithm, so I will not explicitly hardcode a sequence of iterations of this algorithm, but instead provide methods with **kwargs to allow for interoperability and subclassing
    """
    def __init__(self, **kwargs):
        """
        Dummy __init__ method.
        """
        self.iteration = 1 # 1-indexed

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def initialize_particles(self, num_particles = 10, sample_gamma_1 = True, sampler = None, **kwargs):
        """
        Generalized method for Step 1 of SMC algorithm

        args
            num_particles : int
                number of initial particles @ t = N_1
            sample_gamma_1 : bool, default True
                eta_1 = gamma_1 (i.e. we are attempting to pull num_particles i.i.d. samples from the prior);
                if sample_gamma_1, all of the weights are 1 (identically), so we need not compute importance weights
            sampler : generalized_sampler_object
                used to draw i.i.d. samples from the prior

        attributes
            particles : list of apophis.Particle objects
                particles at time t = 1
        """
        from apophis.particles import Particle

        #first, we have to define a target distribution defined at iteration 1
        gamma_1 = self.generate_target_distribution(**kwargs)
        if sample_gamma_1:
            initial_target = gamma_1
        else:
            initial_target = self.generate_distribution(**kwargs)

        #then define the propagator whose invariant is eta_1 and propagate until num_particles i.i.d. samples are rendered
        propagator = self.generate_propagator(sampler = sampler, distribution_object = initial_target, **kwargs)

        self.particles = list()
        for particle in range(num_particles):
            updated_configuration, _ = propagator.propagate(**kwargs)
            weight = 1. if sample_first_target else gamma_1(update_configuration) / initial_target(updated_configuration)
            self.particles.append(Particle(index = particle, configuration = updated_configuration, weight = weight))


    def resample(self, scheme, observable, num_particles, threshold, **kwargs):
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
            resampled_particles = scheme(particles = particles, num_particles = num_particles, **kwargs)
            self.particles = resampled_particles
        else:
            pass

    def update_particle_weights(self, **kwargs):
        """
        Generalized method to compute and update weights for all of the particles
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
            updated_configuration, kernel_density = propagator.propagate(configuration = particle.configuration, **kwargs)
            kernel_densities.update(kernel_density)
            particle.update_configuration(configuration = updated_configuration, **kwargs)

        return np.array(kernel_densities)

    def generate_propagator(self, sampler, distribution_object, **kwargs):
        """
        Generate a propagator from a sampler object and a distribution_object
        """
        propagator = None
        return propagator

    def generate_distribution(self, **kwargs):
        """
        Based on the current iteration, generate and return an object that defines the unnormalized target distribution gamma_n
        """
        target_distribution_object = None
        return target_distribution_object

    def generate_distribution(self, **kwargs):
        """
        generate and return an object that defines some unnormalized distribution
        """
        distribution_object = None
        return distribution_object

    def execute(self, **kwargs):
        """
        Algorithm executor
        """
        #first, initialize_particles
        self.initialize_particles(**kwargs)

        #then, we will attempt to resample...
        self.resample(**kwargs)












class AnnealedImportanceSampling(SequentialMonteCarlo):
    """
    Algorithm 1, developed by R. M. Neal as reported in https://arxiv.org/abs/physics/9803008 (a variant of Sequential Importance Sampling)
    """
    def __init__(self, **kwargs):
        super(AnnealedImportanceSampling, self).__init__(**kwargs)

    def execute(self, **kwargs):
        """

        """
