import numpy as np

import unittest
import Ai

class TestStringMethods(unittest.TestCase):

    def test_find_state(self):
        ai=Ai.Ai()
        x=np.array([1,0,0,1,1,0,0,2,1])

        Ai.find_state(x)