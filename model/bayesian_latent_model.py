import numpy as np
from utils import helper_functions as hf

class BayesianLatentModel:
    def __init__(self, y, q):
        self.y = y
        self.q = q