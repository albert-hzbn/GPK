"""Hardware test for the 3-D CUDA PF/GPK kernels."""
from pathlib import Path
import importlib.util
import sys
import unittest


MODULE = Path(__file__).parents[1] / "scripts" / "gpu_pf_gpk_3d.py"
spec = importlib.util.spec_from_file_location("gpu_pf_gpk_3d", MODULE)
gpu = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gpu
spec.loader.exec_module(gpu)


class GpuPfGpkTests(unittest.TestCase):
    def test_cpu_gpu_parity(self):
        self.assertTrue(gpu.self_test()["passed"])

    def test_all_3d_microstructures_smoke(self):
        for i, (name, generator) in enumerate(gpu.GENERATORS.items()):
            with self.subTest(case=name):
                labels = generator(8, 4, 10 + i)
                self.assertEqual(labels.shape, (8, 8, 8))
                gpk, _, _ = gpu.run_gpk(labels, 2, timed=False)
                pf, _, _ = gpu.run_pf(labels, 4, 2, timed=False)
                self.assertEqual(gpk.shape, labels.shape)
                self.assertEqual(pf.shape, labels.shape)
                self.assertTrue(bool(gpu.cp.all(gpu.cp.isfinite(gpk))))
                self.assertTrue(bool(gpu.cp.all(gpu.cp.isfinite(pf))))


if __name__ == "__main__":
    unittest.main()
