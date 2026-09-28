import unittest
from hs_guard.runtime import serialize
from hs_guard.engine.slot_engine_v7 import pack_inputs
import numpy as np

class ContractTests(unittest.TestCase):
    def test_serialization(self):
        self.assertEqual(serialize([{'role':'user','content':'Hello'},{'role':'assistant','content':'Hi'}]),'USER:\nHello\n\nASSISTANT:\nHi')
    def test_empty_target(self):
        with self.assertRaises(ValueError):serialize([{'role':'user','content':''}])
    def test_padding_uses_scratch_slot(self):
        out=np.empty(2*4+3*2,dtype=np.int64)
        lens=pack_inputs([[4,5,6]],[0],2,4,np.array([7]),1,out)
        self.assertEqual(lens.tolist(),[3]);self.assertEqual(out[:8].tolist(),[4,5,6,0,0,0,0,0])
        self.assertEqual(out[8:10].tolist(),[3,0]);self.assertEqual(out[10:12].tolist(),[0,1]);self.assertEqual(out[12:].tolist(),[7,0])

if __name__=='__main__':unittest.main()
