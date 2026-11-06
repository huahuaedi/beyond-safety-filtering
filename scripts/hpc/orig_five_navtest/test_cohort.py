import unittest
from prepare import validate_cohort
class CohortTest(unittest.TestCase):
 def setUp(self):
  self.t=[str(i) for i in range(1000)];self.m={t:str(i%136) for i,t in enumerate(self.t)};self.r=[{'token':t} for t in self.t]
 def test_valid(self):validate_cohort(self.t,self.m,self.r,self.t)
 def test_order_mismatch(self):
  with self.assertRaises(ValueError):validate_cohort(self.t,self.m,self.r,list(reversed(self.t)))
 def test_duplicate(self):
  t=self.t[:-1]+[self.t[0]]
  with self.assertRaises(ValueError):validate_cohort(t,self.m,self.r,t)
 def test_log_mismatch(self):
  with self.assertRaises(ValueError):validate_cohort(self.t,{t:'same' for t in self.t},self.r,self.t)
if __name__=='__main__':unittest.main()
