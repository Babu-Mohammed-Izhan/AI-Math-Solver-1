import unittest
from ai import *
import json


class TestSum(unittest.TestCase):

    def test_not_ai(self):
        self.assertEqual(ai(json.loads('{ "message": "This is not a math equation", "history" : [] }')), "Sorry, this is a invalid question. How can I help you with math?", "Should be a invalid math equation")

if __name__ == '__main__':
    unittest.main()