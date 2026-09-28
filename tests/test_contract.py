import unittest
import json,hashlib,tempfile
from pathlib import Path
from hs_guard.runtime import verify_model
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

class ManifestTests(unittest.TestCase):
    def test_hub_snapshot_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'snapshots'/'revision';blobs=base/'blobs'
            root.mkdir(parents=True);blobs.mkdir()
            files={}
            for name in ['config.json','tokenizer.json','model.safetensors']:
                content=name.encode();digest=hashlib.sha256(content).hexdigest()
                (blobs/digest).write_bytes(content);(root/name).symlink_to('../../blobs/'+digest);files[name]=digest
            manifest={'format':'hs-guard-v1','risk_labels':['safe','unsafe','controversial'],'files':files}
            (root/'hs_guard.json').write_text(json.dumps(manifest))
            self.assertEqual(verify_model(root)['files'],files)
            manifest['files']['../escape']='0'*64
            (root/'hs_guard.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_model(root)
    def test_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);root=base/'bundle';root.mkdir();files={}
            for name in ['config.json','tokenizer.json','model.safetensors']:
                (base/name).write_text(name);(root/name).symlink_to(base/name);files[name]=hashlib.sha256(name.encode()).hexdigest()
            (root/'hs_guard.json').write_text(json.dumps({'format':'hs-guard-v1','risk_labels':['safe','unsafe','controversial'],'files':files}))
            with self.assertRaises(ValueError):verify_model(root)

if __name__=='__main__':unittest.main()
