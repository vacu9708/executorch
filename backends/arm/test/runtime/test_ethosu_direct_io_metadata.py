# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import struct
import sys

from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from executorch.backends.arm.arm_vela import has_vela
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.test.tester.arm_tester import ArmTester
from executorch.backends.test.harness.stages import StageType
from executorch.exir.lowered_backend_module import LoweredBackendModule

pytestmark = pytest.mark.skipif(
    not has_vela, reason="ethos-u-vela is required for Ethos-U lowering"
)


class _ReluAdd(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + 1.0)


def _vela_blocks(blob: bytes) -> dict[str, bytes]:
    blocks: dict[str, bytes] = {}
    offset = 0
    while offset + 32 <= len(blob):
        name = blob[offset : offset + 16].split(b"\0", 1)[0].decode()
        block_size = struct.unpack_from("<i", blob, offset + 16)[0]
        data_start = offset + 32
        data_end = data_start + block_size
        assert data_end <= len(blob)
        blocks[name] = blob[data_start:data_end]
        offset = data_start + ((block_size + 15) // 16) * 16
        if name == "vela_end_stream":
            break
    return blocks


def _vela_io_regions(block: bytes) -> list[tuple[int, int]]:
    count = struct.unpack_from("<i", block, 0)[0]
    result: list[tuple[int, int]] = []
    offset = 4
    for _ in range(count):
        values = struct.unpack_from("<iiiiiiiii", block, offset)
        result.append((values[7], values[8]))
        offset += 36
    return result


def _lowered_vela_blocks(extra_flags: list[str]) -> dict[str, bytes]:
    compile_spec = EthosUCompileSpec(
        "ethos-u85-128",
        extra_flags=extra_flags,
        system_config="Ethos_U85_SYS_DRAM_Mid",
        memory_mode="Shared_Sram",
    )
    tester = ArmTester(
        _ReluAdd(),
        example_inputs=(torch.randn(1, 8, 8, 8),),
        compile_spec=compile_spec,
    )
    tester.quantize().export().to_edge_transform_and_lower()

    graph_module = tester.stages[StageType.TO_EDGE_TRANSFORM_AND_LOWER].graph_module
    for node in graph_module.graph.nodes:
        if node.op == "get_attr" and node.name.startswith("lowered_module_"):
            lowered_module = getattr(graph_module, node.name)
            assert isinstance(lowered_module, LoweredBackendModule)
            return _vela_blocks(lowered_module.processed_bytes)
    raise AssertionError("Expected an Ethos-U lowered module")


def test_separate_io_regions_are_preserved_in_vela_bin_stream() -> None:
    default_blocks = _lowered_vela_blocks([])
    direct_blocks = _lowered_vela_blocks(["--separate-io-regions", "--cop-format=COP2"])

    assert _vela_io_regions(default_blocks["inputs"]) == [(0, 1)]
    assert _vela_io_regions(default_blocks["outputs"]) == [(0, 1)]
    assert _vela_io_regions(direct_blocks["inputs"]) == [(0, 3)]
    assert _vela_io_regions(direct_blocks["outputs"]) == [(0, 4)]
