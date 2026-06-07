/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Arm backend for Ethos-U baremetal driver stack, this relies on the
 * ethos-u-core-driver for hardware interaction.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include <ethosu_driver.h>

#include <executorch/backends/arm/runtime/EthosUBackend_Internal.h>
#include <executorch/runtime/core/error.h>

using executorch::runtime::BackendExecutionContext;
using executorch::runtime::Error;
using executorch::runtime::Span;

// Compatibility hooks for multi-device driver / non-multi-device driver code
// When multi-device driver code is available, these declarations are overridden
extern "C" __attribute__((weak)) int ethosu_get_product_config_from_cop_data(
    const void*,
    const int,
    uint32_t* product_out,
    uint32_t* log2_macs_out) {
  *product_out = 0;
  *log2_macs_out = 0;
  return 0;
}

extern "C" __attribute__((weak)) struct ethosu_driver* ethosu_reserve_driver_ex(
    uint32_t,
    uint32_t) {
  return ethosu_reserve_driver();
}

// Overridable memcpy used by the EthosU backend for output scratch
// shuffling. Default (weak) implementation in EthosUBackend_IoMemcpy.cpp does
// std::memcpy. Firmware targets can supply a strong override (e.g. routing
// through a DMA engine) to reduce CPU memcpy load on the host MCU.
extern "C" void arm_ethos_io_memcpy(void* dst, const void* src, size_t size);

namespace executorch {
namespace backends {
namespace arm {

struct PlatformState {};

namespace {

// Sets a base address for a direct IO region,
// checking that the provided tensor data pointer and offset are consistent
Error set_direct_io_base(
    const char* io_name,
    int io_index,
    int region,
    uint64_t* bases,
    size_t* bases_size,
    bool* bases_set,
    int* num_base_addr,
    const void* data,
    size_t data_size,
    int offset) {
  if (region < 0 || region >= kEthosUMaxBaseAddrCount) {
    ET_LOG(
        Error,
        "%s %d uses unsupported Vela region %d",
        io_name,
        io_index,
        region);
    return Error::InvalidProgram;
  }
  if (offset < 0) {
    ET_LOG(Error, "%s %d has negative offset %d", io_name, io_index, offset);
    return Error::InvalidProgram;
  }
  const size_t region_offset = static_cast<size_t>(offset);
  if (data_size > SIZE_MAX - region_offset) {
    ET_LOG(Error, "%s %d direct IO range overflows", io_name, io_index);
    return Error::InvalidProgram;
  }

  const uintptr_t data_addr = reinterpret_cast<uintptr_t>(data);
  if (data_addr < region_offset) {
    ET_LOG(
        Error,
        "%s %d direct IO offset exceeds tensor address",
        io_name,
        io_index);
    return Error::InvalidProgram;
  }

  const uintptr_t base_addr = data_addr - region_offset;
  if ((base_addr & 0xFUL) != 0) {
    ET_LOG(
        Error,
        "%s %d direct IO base is not 16-byte aligned",
        io_name,
        io_index);
    return Error::InvalidProgram;
  }

  const uint64_t base = static_cast<uint64_t>(base_addr);
  const size_t required_size = region_offset + data_size;
  if (!bases_set[region]) {
    bases[region] = base;
    bases_size[region] = required_size;
    bases_set[region] = true;
  } else if (bases[region] != base) {
    ET_LOG(
        Error,
        "%s %d is not contiguous with other tensors in Vela region %d",
        io_name,
        io_index,
        region);
    return Error::InvalidProgram;
  } else if (required_size > bases_size[region]) {
    bases_size[region] = required_size;
  }

  if (*num_base_addr < region + 1) {
    *num_base_addr = region + 1;
  }
  return Error::Ok;
}

} // namespace

PlatformState* platform_init(
    executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> /*specs*/,
    executorch::runtime::MemoryAllocator* /*allocator*/) {
  return nullptr;
}

void platform_destroy(PlatformState* state) {
  delete state;
}

Error platform_execute(
    BackendExecutionContext& /*context*/,
    const ExecutionHandle* /*execution_handle*/,
    const VelaHandles& handles,
    int input_count,
    int output_count,
    Span<executorch::runtime::EValue*> args,
    char* ethosu_scratch) {
  // Parse product config from command stream to reserve the correct driver
  uint32_t product, log2_macs;
  // The weak fallback below always returns 0, but some builds replace it
  // with a real driver implementation that can return an error code.
  const int product_config_status = ethosu_get_product_config_from_cop_data(
      handles.cmd_data, handles.cmd_data_size, &product, &log2_macs);
  if (product_config_status != 0) { // cppcheck-suppress knownConditionTrueFalse
    ET_LOG(Error, "Failed to parse product config from command stream");
    return Error::InvalidProgram;
  }

  // Allocate driver handle and synchronously invoke driver
  auto driver =
      std::unique_ptr<ethosu_driver, decltype(&ethosu_release_driver)>(
          ethosu_reserve_driver_ex(product, log2_macs), ethosu_release_driver);
  if (driver == nullptr) {
    ET_LOG(Error, "ethosu_reserve_driver_ex failed");
    return Error::InvalidState;
  }

  uint64_t bases[kEthosUMaxBaseAddrCount] = {};
  size_t bases_size[kEthosUMaxBaseAddrCount] = {};
  bool bases_set[kEthosUMaxBaseAddrCount] = {};
  bases[kVelaWeightRegion] =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handles.weight_data));
  bases_size[kVelaWeightRegion] = handles.weight_data_size;
  bases_set[kVelaWeightRegion] = true;
  bases[kVelaScratchRegion] =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_scratch));
  bases_size[kVelaScratchRegion] = handles.scratch_data_size;
  bases_set[kVelaScratchRegion] = true;
  bases[kVelaFastScratchRegion] =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ethosu_fast_scratch));
  bases_size[kVelaFastScratchRegion] = ethosu_fast_scratch_size;
  bases_set[kVelaFastScratchRegion] = true;

  int num_base_addr = kEthosUDefaultBaseAddrCount;

  for (int i = 0; i < input_count; ++i) {
    const VelaIO& input_io = handles.inputs->io[i];
    if (input_io.region == kVelaScratchRegion) {
      continue;
    }
    if (input_io.region != kVelaInputRegion) {
      ET_LOG(
          Error,
          "Input %d uses unsupported Vela region %d",
          i,
          input_io.region);
      return Error::InvalidProgram;
    }

    auto tensor_in = args[i]->toTensor();
    Error status = set_direct_io_base(
        "Input",
        i,
        input_io.region,
        bases,
        bases_size,
        bases_set,
        &num_base_addr,
        tensor_in.const_data_ptr<char>(),
        tensor_in.nbytes(),
        input_io.offset);
    if (status != Error::Ok) {
      return status;
    }
  }

  for (int i = 0; i < output_count; ++i) {
    VelaIO& output_io = handles.outputs->io[i];
    if (output_io.region == kVelaScratchRegion) {
      continue;
    }
    if (output_io.region != kVelaOutputRegion) {
      ET_LOG(
          Error,
          "Output %d uses unsupported Vela region %d",
          i,
          output_io.region);
      return Error::InvalidProgram;
    }

    int tensor_count = 1, io_count = 1;
    auto tensor_out = args[input_count + i]->toTensor();
    calculate_dimensions(tensor_out, &output_io, &tensor_count, &io_count);
    const size_t tensor_bytes = tensor_out.nbytes();
    const size_t io_bytes = static_cast<size_t>(io_count) *
        static_cast<size_t>(output_io.elem_size);
    if (tensor_bytes != io_bytes) {
      ET_LOG(
          Error,
          "Output %d with direct Vela region %d requires layout adjustment",
          i,
          output_io.region);
      return Error::InvalidProgram;
    }

    Error status = set_direct_io_base(
        "Output",
        i,
        output_io.region,
        bases,
        bases_size,
        bases_set,
        &num_base_addr,
        tensor_out.mutable_data_ptr<char>(),
        tensor_bytes,
        output_io.offset);
    if (status != Error::Ok) {
      return status;
    }
  }

  int result = ethosu_invoke_v3(
      driver.get(),
      static_cast<const void*>(handles.cmd_data),
      handles.cmd_data_size,
      bases,
      bases_size,
      num_base_addr,
      nullptr);

  if (result != 0) {
    ET_LOG(Error, "Ethos-U invocation failed error (%d)", result);
    return Error::InvalidProgram;
  }

  size_t tensor_bytes_total = 0;
  size_t io_bytes_total = 0;
  // Scratch-region outputs are copied into EValue tensors; direct outputs have
  // already been written there by the driver.
  for (int i = 0; i < output_count; i++) {
    int tensor_count = 1, io_count = 1;
    VelaIO& output_io = handles.outputs->io[i];
    // Outputs are in the index immediately after inputs.
    auto tensor_out = args[input_count + i]->toTensor();

    calculate_dimensions(tensor_out, &output_io, &tensor_count, &io_count);

    size_t tensor_bytes = tensor_out.nbytes();
    size_t io_bytes = static_cast<size_t>(io_count) *
        static_cast<size_t>(output_io.elem_size);

    if (output_io.region == kVelaOutputRegion) {
      if (tensor_bytes != io_bytes) {
        ET_LOG(Error, "Output tensor sizes do not match");
        return Error::InvalidProgram;
      }
      io_bytes_total += io_bytes;
    } else if (output_io.region == kVelaScratchRegion) {
      if (output_io.offset < 0) {
        ET_LOG(
            Error,
            "Output %d has negative scratch offset %d",
            i,
            output_io.offset);
        return Error::InvalidProgram;
      }
      const size_t output_offset = static_cast<size_t>(output_io.offset);
      if (output_offset > handles.scratch_data_size ||
          io_bytes > handles.scratch_data_size - output_offset) {
        ET_LOG(Error, "Output %d scratch range exceeds scratch buffer size", i);
        return Error::InvalidProgram;
      }
      const char* output_addr = ethosu_scratch + output_offset;
      if (tensor_bytes != io_bytes) {
        Error status = copy_with_layout_adjustment(
            output_io, i, output_addr, tensor_out, tensor_bytes);
        if (status != Error::Ok) {
          return status;
        }
        io_bytes_total += tensor_bytes;
      } else {
        // Routed through arm_ethos_io_memcpy so firmware can DMA-accelerate.
        arm_ethos_io_memcpy(
            tensor_out.mutable_data_ptr<char>(),
            static_cast<const char*>(output_addr),
            tensor_bytes);
        io_bytes_total += io_bytes;
      }
    } else {
      ET_LOG(
          Error,
          "Output %d uses unsupported Vela region %d",
          i,
          output_io.region);
      return Error::InvalidProgram;
    }

    // At times the topological order of the outputs may change.
    // Lets instead ensure that the sum of output bytes match.
    tensor_bytes_total += tensor_bytes;
  }
  if (tensor_bytes_total != io_bytes_total) {
    ET_LOG(Error, "Total output tensor sizes do not match");
    ET_LOG(
        Error,
        "Program expects %zu bytes but got %zu",
        io_bytes_total,
        tensor_bytes_total);
    return Error::InvalidProgram;
  }
  return Error::Ok;
}

} // namespace arm
} // namespace backends
} // namespace executorch
