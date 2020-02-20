"""Propagator Module"""
#####Imports#####
import numpy as np
import copy

class Propagator(object):
    """
    Dummy super to wrap the functionality of various classes that define propagators.
    """
    def __init__(self, **kwargs):
        """
        Dummy init method.
        """
        #update other kwargs that have not been set
        self.__dict__.update(kwargs)
