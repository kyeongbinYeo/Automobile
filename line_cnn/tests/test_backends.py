"""Backend selection and the ONNX tag that stops the wrong weights from loading silently."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import IN_H, IN_W  # noqa: E402
from perception import (OnnxBackend, TwinLiteNetBackend, TwinLiteOnnxBackend,  # noqa: E402
                        create_backend)
from pipeline import load_config  # noqa: E402

W = os.path.join(HERE, "..", "weights")
LINENET = os.path.join(W, "linenet.onnx")
TWIN = os.path.join(W, "twinlite_ft.onnx")
TWIN_PT = os.path.join(W, "twinlite_ft.pt")
REPO = os.path.join(HERE, "..", "..", "third_party", "TwinLiteNet")
CFG = load_config(os.path.join(HERE, "..", "configs", "default.yaml"))["perception"]
HAVE = os.path.exists(LINENET) and os.path.exists(TWIN)


def img(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (IN_H, IN_W, 3), dtype=np.uint8)


@unittest.skipUnless(HAVE, "exported weights not present")
class BackendSelectionTests(unittest.TestCase):
    def test_default_is_linenet(self):
        self.assertEqual(CFG["backend"], "linenet")
        self.assertIsInstance(create_backend(CFG, LINENET), OnnxBackend)

    def test_twinlite_onnx(self):
        b = create_backend({**CFG, "backend": "twinlite"}, TWIN)
        self.assertIsInstance(b, TwinLiteOnnxBackend)
        ll, da = b.predict(img())
        self.assertEqual(ll.shape, (IN_H, IN_W))
        self.assertTrue(0.0 <= float(ll.min()) and float(ll.max()) <= 1.0)

    def test_wrong_weights_are_refused_in_both_directions(self):
        with self.assertRaises(ValueError):
            create_backend({**CFG, "backend": "twinlite"}, LINENET)
        with self.assertRaises(ValueError):
            create_backend({**CFG, "backend": "linenet"}, TWIN)

    def test_unknown_backend(self):
        with self.assertRaises(ValueError):
            create_backend({**CFG, "backend": "nope"}, LINENET)

    def test_models_really_differ(self):
        a = create_backend(CFG, LINENET).predict(img(3))[0]
        b = create_backend({**CFG, "backend": "twinlite"}, TWIN).predict(img(3))[0]
        self.assertGreater(float(np.abs(a - b).mean()), 1e-3)


@unittest.skipUnless(HAVE and os.path.exists(TWIN_PT) and os.path.exists(os.path.join(REPO, "model", "TwinLite.py")),
                     "needs the TwinLiteNet checkout and tuned weights")
class TwinLiteExportTests(unittest.TestCase):
    def test_onnx_matches_pytorch(self):
        onnx_b = TwinLiteOnnxBackend(TWIN, use_gamma=True)
        torch_b = TwinLiteNetBackend.from_repo(REPO, TWIN_PT, size=(IN_W, IN_H), use_gamma=True)
        x = img(5)
        for a, b in zip(onnx_b.predict(x), torch_b.predict(x)):
            self.assertLess(float(np.abs(a - b).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()
