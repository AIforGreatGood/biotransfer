# Copyright (c) 2021 Massachusetts Institute of Technology
# Subject to FAR 52.227-11 – Patent Rights – Ownership by the Contractor (May 2014).

# This file is a copy of various support functions in  https://github.com/samsinai/FLEXS, release version 0.2.8. 
# MIT Lincoln Laboratory has adapted these support functions to enable a cohesive design process without the need
# to install additional packages that FLEXS requires. 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from typing import Any, List, Union
import numpy as np
import abc

SEQUENCES_TYPE = Union[List[str], np.ndarray]

class Landscape(abc.ABC):
    """
    Base class for all landscapes and for `flexs.Model`.
    Attributes:
        cost (int): Number of sequences whose fitness has been evaluated.
        name (str): A human-readable name for the landscape (often contains
            parameter values in the name) which is used when logging explorer runs.
    """

    def __init__(self, name: str):
        """Create Landscape, setting `name` and setting `cost` to zero."""
        self.cost = 0
        self.name = name

    @abc.abstractmethod
    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        pass

    def get_fitness(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        """
        Score a list/numpy array of sequences.
        This public method should not be overriden – new landscapes should
        override the private `_fitness_function` method instead. This method
        increments `self.cost` and then calls and returns `_fitness_function`.
        Args:
            sequences: A list/numpy array of sequence strings to be scored.
        Returns:
            Scores for each sequence.
        """
        self.cost += len(sequences)
        return self._fitness_function(sequences)

class Model(Landscape, abc.ABC):
    """
    Base model class. Inherits from `flexs.Landscape` and adds an additional
    `train` method.
    """

    @abc.abstractmethod
    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """
        Train model.
        This function is called whenever you would want your model to update itself
        based on the set of sequences it has measurements for.
        """
        pass


class LandscapeAsModel(Model):
    """
    This simple class wraps a `flexs.Landscape` in a `flexs.Model` to allow running
    experiments against a perfect model.
    This class's `_fitness_function` simply calls the landscape's `_fitness_function`.
    """

    def __init__(self, landscape: Landscape):
        """
        Create a `flexs.Model` out of a `flexs.Landscape`.
        Args:
            landscape: Landscape to wrap in a model.
        """
        super().__init__(f"LandscapeAsModel={landscape.name}")
        self.landscape = landscape

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        return self.landscape._fitness_function(sequences)

    def train(self, sequences: SEQUENCES_TYPE, labels: List[Any]):
        """No-op."""
        pass