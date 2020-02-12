"""Particle Module"""
#####Imports#####
import numpy as np

class Particle(object):
    """
    Generalized Particle object

    The Particle Object holds particle-specific information like ancestry, configuration, weight, etc.
    """
    def __init__(self, index = 0, configuration = None, weight = None, **kwargs):
        """
        Dummy init method to be overwritten.

        args
            index : int
                index label of particle
            configuration : generalized configuration object
                configuration of parameters
            weight : float
                unnormalized weight of particle
        """
        self.index = index
        self.weight = weight
        self.configuration = configuration

        self.ancestry = [index]
        self.configurations = [configuration]
        self.weights = [weight]

        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def update_configuration(self, configuration, record_history = True, **kwargs):
        """
        Wrapper method to update configuration

        args
            configuration : generalized configuration object
                the configuration of the particle for which we will update
            record_history : bool, default True
                whether to add the configuration to the history
        """
        self.configuration = configuration

        if record_history:
            self.configurations.append(configuration)


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

    def update_weight(self, weight, record_weight = True):
        """
        Generalized method to update the weight of a particle

        args
            weight : float
                weight to update
            record_weight : bool, default True
                whether to record the weight history
        """
        self.weight = weight
        if record_weight:
            self.weights.append(weight)
