from pathlib import Path
import sys,json,tempfile,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'packages/pi_default/runtime'))
from verify_bundle import verify

class PackagingTests(unittest.TestCase):
    def test_empty_bundle_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps(dict(models=0,contracts=[],sha256={})))
            with self.assertRaisesRegex(ValueError,'nonempty'):verify(p)
    def test_platform_specific_manifest_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps(dict(models=1,contracts=['onnx\\contract.json'],sha256={})))
            with self.assertRaisesRegex(ValueError,'forward slashes'):verify(p)
    def test_missing_hash_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text(json.dumps(dict(models=1,contracts=['contract.json'],sha256={})))
            with self.assertRaisesRegex(ValueError,'Unhashed'):verify(p)

if __name__=='__main__':unittest.main(verbosity=2)
