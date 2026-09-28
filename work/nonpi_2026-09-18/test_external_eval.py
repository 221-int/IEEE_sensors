import unittest
from external_eval import match,intervals
class MatchingTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(intervals([1,1,0,1]),[(0,1),(3,3)])
        self.assertEqual(intervals([]),[])
    def test_no_multiple_credit(self):
        self.assertEqual(len(match([(0,20)],[(1,4),(7,10)])),1)
        self.assertEqual(len(match([(1,2),(3,4)],[(0,10)])),1)
    def test_augmenting_path(self):
        self.assertEqual(len(match([(0,12),(0,3)],[(0,3),(10,12)])),2)
    def test_iou_boundary(self):
        self.assertEqual(len(match([(0,99)],[(0,9)],.1)),1)
        self.assertEqual(len(match([(0,100)],[(0,9)],.1)),0)
if __name__=='__main__':unittest.main()
