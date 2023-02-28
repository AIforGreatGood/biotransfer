# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

from src.flexs_modules import Landscape
import numpy as np
from torch import Tensor

class ProxyLandscape(Landscape):
    """A landscape for evaluating on the proxy problem"""

    def __init__(self, model, maximize=True, exp=False):
        super().__init__(name="ProxyLandscape")

        self.model = model
        self.maximize = maximize
        self.exp = exp

    def _fitness_function(self, sequences):

        fitnesses = self.model.predict(sequences)
        if type(fitnesses) is Tensor:
            fitnesses = fitnesses.detach().cpu().numpy()

        fitnesses = np.asarray(fitnesses)
        if len(fitnesses.shape) == 0:
            fitnesses = np.expand_dims(fitnesses, axis=0)

        if not self.maximize:
            fitnesses *= -1

        # This eliminates negative values from the fitness function, which throws issues with the genetic algorithm.
        # Open to change, this was just an easy GA fix.
        if self.exp:
            fitnesses = np.exp(fitnesses)

        return fitnesses

    def train(*args, **kwargs):
        pass