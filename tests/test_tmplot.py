import unittest
import pickle as pkl
from tomotopy import LDAModel
from src import tmplot as tm


class TestTmplot(unittest.TestCase):

    tomotopy_model = LDAModel.load('tests/models/tomotopyLDA.model')
    with open('tests/models/gensimLDA.model', 'rb') as file:
        gensim_model = pkl.load(file)
    with open('tests/models/gensimLDA.corpus', 'rb') as file:
        gensim_corpus = pkl.load(file)

    def test_get_phi(self):
        phi = tm.get_phi(self.tomotopy_model)
        phi2 = tm.get_phi(self.gensim_model)
        self.assertGreater(phi.shape[0], 0)
        self.assertGreater(phi2.shape[0], 0)
        self.assertEqual(phi.shape[1], 15)
        self.assertEqual(phi2.shape[1], 15)

    def test_get_theta(self):
        theta = tm.get_theta(self.tomotopy_model)
        theta2 = tm.get_theta(self.gensim_model, self.gensim_corpus)
        self.assertEqual(theta.shape[0], 15)
        self.assertEqual(theta2.shape[0], 15)
        self.assertGreater(theta.shape[1], 0)
        self.assertGreater(theta2.shape[1], 0)

    def test_prepare_coords(self):
        coords = tm.prepare_coords(self.tomotopy_model)
        self.assertTupleEqual(coords.shape, (self.tomotopy_model.k, 5))

    def test_is_tomotopy(self):
        self.assertTrue(tm._helpers._is_tomotopy(self.tomotopy_model))

    def test_is_gensim(self):
        self.assertTrue(tm._helpers._is_gensim(self.gensim_model))


if __name__ == '__main__':
    unittest.main()
