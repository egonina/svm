import unittest2 as unittest
import copy
import numpy as np
from svm_specializer.svm import * 

class BasicTests(unittest.TestCase):
    def test_init(self):
        svm = SVM()
        self.assertIsNotNone(svm)

class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        feats = open("svm_test.svm", "r")
        labels = []
        points = {}
        self.D = 0
        first_line = 1

        for line in feats:
            vals = line.split(" ")
            l = vals[0]
            labels.append(l)
            idx = 0
            for v in vals[1:]:
                if first_line:
                    self.D += 1
                f = v.split(":")[1].strip('\n')
                if idx not in points.keys():
                    points[idx] = []
                points[idx].append(f)
                idx += 1
            if first_line:
                first_line = 0

        self.N = len(labels)
        self.labels = np.array(labels, dtype=np.float32)
        points_list  = [] 

        for idx in points.keys():
           points_list.append(points[idx]) 

        self.points = np.array(points_list, dtype=np.float32)
        self.points = self.points.reshape(self.N, self.D)

    
    def test_training_once(self):
        svm = SVM()
        a = svm.train(self.points, self.labels, "linear")

    def test_training_and_classify_once(self):
        svm = SVM()
        a = svm.train(self.points, self.labels, "linear")
        a = svm.classify(self.points, self.labels)

if __name__ == '__main__':
    unittest.main()
