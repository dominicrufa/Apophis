"""Thermostate Module"""
#####Imports#####
import numpy as np
import copy

class ThermodynamicState(object):
    """
    Dummy super to wrap the functionality of various classes that define invariant distributions
    ThermodynamicStates define invariants of target distributions and proposal kernels.
    """
    def __init__(self, **kwargs):
        """
        Dummy init method.
        """
        #update other kwargs that have not been set
        self.__dict__.update(kwargs)

    def set_parameters(self, parameters, **kwargs):
        """
        Dummy wrapper to set parameters to define ThermodynamicState invariants
        args
            parameters : np.array or float
                parameters defining new invariant thermodynamic_state
        """
        pass

class PersesCompoundAlchemicalThermodynamicState(ThermodynamicState):
    """
    Wrapper subclass for openmmtools.states.CompoundThermodynamicState object as defined by Perses
    """
    def __init__(self, thermodynamic_state, **kwargs):
        """
        Wrap an openmmtools thermodynamic_state or CompoundThermodynamicState
        """
        from perses.annihilation.lambda_protocol import LambdaProtocol
        self.thermodynamic_state = copy.deepcopy(thermodynamic_state)
        super(OpenMMToolsThermodynamicState, self).__init__(**kwargs)

    def set_parameters(self, parameters, lambda_protocol = LambdaProtocol(), **kwargs):
        """
        wrapper method to update parameters
        args
            parameters : np.array or float
                parameters defining new invariant thermodynamic_state
            lambda_protocol : perses.annihilation.lambda_protocol.LambdaProtocol
        """
        self.thermodynamic_state.set_alchemical_parameters(parameters, lambda_protocol)
