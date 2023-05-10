import unittest
from ai import *
import json


class TestSum(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(ai(json.loads('{ "message": "This is not a math equation", "history" : [] }')), "Sorry, this is a invalid question. How can I help you with math?", "Should be a invalid math equation")

    def test_sum_tuple(self):
        self.assertIn("[[4, 5, 7], [6, 9, 11]]",ai(json.loads('{ "message": "Add [[2,5,7], [4,7,9]] and [[2,0,0], [2,2,2]]", "history" : [] }')), "Should be able to solve the maths equation")

if __name__ == '__main__':
    unittest.main()