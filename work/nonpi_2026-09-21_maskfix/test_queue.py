import tempfile,unittest,json,os
from pathlib import Path
from unittest.mock import patch
import experiment_queue as Q

class PreservationTests(unittest.TestCase):
    def test_existing_copy_must_match(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b';a.write_bytes(b'original')
            Q.copy_once(a,b);Q.copy_once(a,b)
            a.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):Q.copy_once(a,b)
            self.assertEqual(b.read_bytes(),b'original')
    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.json';Q.write(p,{'complete':6})
            self.assertEqual(Q.read(p),{'complete':6})
            self.assertFalse(p.with_suffix('.json.tmp').exists())
    def test_full_comparison_refuses_partial(self):
        with patch.object(Q,'load_variant',return_value={'runs':{(0,0):{}},'meta':{}}):
            with self.assertRaisesRegex(ValueError,'Full 15-run'):Q.compare_all()
    def test_live_process_detection(self):
        self.assertTrue(Q.process_alive(os.getpid()))
        self.assertFalse(Q.process_alive(0))
if __name__=='__main__':unittest.main()
