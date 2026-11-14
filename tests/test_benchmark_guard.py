"""Test that test evaluation cannot start before a valid dev selection."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'starter'/'retrieval_benchmark.py'


class BenchmarkGuardTests(unittest.TestCase):
    def run_test_split(self, directory):
        return subprocess.run([sys.executable,str(SCRIPT),'--split','test','--output',str(directory)],
                              text=True,capture_output=True,check=False)

    def test_no_test_run_before_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.run_test_split(Path(temp))
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Record the selected dev configuration',result.stderr)

    def test_changed_benchmark_invalidates_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'selection.json').write_text(json.dumps({'benchmark_sha256':'incorrect'}),encoding='utf8')
            result = self.run_test_split(root)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Benchmark changed after configuration selection',result.stderr)


if __name__ == '__main__': unittest.main()
