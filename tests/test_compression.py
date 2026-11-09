
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

# Import existing project modules.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from pruning import prune_mlp
from quantization import convert_conv1d


class TinyTransformerBlock(nn.Module):
    """Small feed-forward block for testing."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()

        self.mlp.c_fc = Conv1D(16, 8)
        self.mlp.c_proj = Conv1D(8, 16)

    def forward(self, x):
        x = self.mlp.c_fc(x)
        x = torch.nn.functional.gelu(x)
        return self.mlp.c_proj(x)


class TestCompression(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_structured_pruning(self):
        """Pruning should reduce parameters."""

        block = TinyTransformerBlock()
        inputs = torch.randn(2, 4, 8)

        original_output = block(inputs)

        original_params = sum(
            p.numel() for p in block.parameters()
        )

        retained = prune_mlp(
            block,
            pruning_rate=0.25
        )

        compressed_output = block(inputs)

        compressed_params = sum(
            p.numel() for p in block.parameters()
        )

        self.assertEqual(retained, 12)

        self.assertLess(
            compressed_params,
            original_params
        )

        self.assertEqual(
            original_output.shape,
            compressed_output.shape
        )

        self.assertTrue(
            torch.isfinite(compressed_output).all().item()
        )

    def test_conv1d_conversion(self):
        """Conversion should preserve FP32 output."""

        model = nn.Sequential(
            Conv1D(7, 5),
            Conv1D(3, 7),
        ).eval()

        inputs = torch.randn(2, 4, 5)

        with torch.inference_mode():
            original = model(inputs)

        convert_conv1d(model)

        with torch.inference_mode():
            converted = model(inputs)

        self.assertIsInstance(
            model[0], nn.Linear
        )

        self.assertIsInstance(
            model[1], nn.Linear
        )

        torch.testing.assert_close(
            original,
            converted,
            rtol=1e-5,
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
