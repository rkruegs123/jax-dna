import pdb
import unittest

from jax_dna.dna1.load_params import load


BASE_PARAMS = load(process=False)

class Dna1Model:
    def __init__(self):
        pass


class TestDna1Model(unittest.TestCase):

    def test_init(self):
        model = Dna1Model()

if __name__ == "__main__":
    unittest.main()
