"""Check accumulation tails, strict scene levels, and live/stale run-lock behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from datasets.common import parse_mill_filename
from utils.experiment import RunLock
from utils.runtime import PROJECT_ROOT
from utils.training import accumulation_samples


class ExperimentTests(unittest.TestCase):
    def test_negative_level_parser_is_strict(self):
        self.assertEqual(parse_mill_filename("Scene1_a-10.png"), ("Scene1_a", 10.0))
        for filename in ("Scene1_a-0.png", "Scene1_a-11.png", "Scene1_a_3.png", "Scene1_a-3.0.png"):
            with self.assertRaises(ValueError):
                parse_mill_filename(filename)

    def test_weighted_accumulation_matches_group_means_with_short_tail(self):
        # N=9, batch=2, accumulation=2 gives groups of 4,4,1 actual samples.
        features = torch.arange(1, 10, dtype=torch.float32)
        actual = torch.tensor(0.2, requires_grad=True)
        reference = actual.detach().clone().requires_grad_(True)
        for start in (0, 4, 8):
            group = features[start:start+4]
            ((reference * group).square().mean()).backward()
            expected_gradient = reference.grad.clone()
            reference.grad = None
            for offset in range(start, min(start+4, len(features)), 2):
                batch = features[offset:offset+2]
                samples = accumulation_samples(9, 2, 2, offset // 2)
                ((actual * batch).square().mean() * len(batch) / samples).backward()
            torch.testing.assert_close(actual.grad, expected_gradient)
            actual.grad = None

    def test_live_lock_rejects_second_process_and_stale_cleanup_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as name:
            directory = Path(name)
            with RunLock(directory):
                with self.assertRaises(RuntimeError):
                    with RunLock(directory):
                        pass
            path = directory / "run.lock"
            path.write_text(json.dumps({"pid": 2147483646, "created": 0}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                with RunLock(directory):
                    pass
            with RunLock(directory, clear_stale=True):
                self.assertTrue(path.exists())
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
