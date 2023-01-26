#include "rmm/detail/cuda_util.hpp"
#include <catch2/catch.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace {

__global__ void kernel(int* input, int* output, int n, int iterations = 1)
{
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    for (int i = 0; i < iterations; i++) {  // so we can control runtime
      output[idx] = input[idx];
    }
  }
}

}  // namespace

TEST_CASE("device_buffer use-after-free a", "[example_3a]")
{
  auto async_mr = rmm::mr::cuda_async_memory_resource{15UL << 30};
  rmm::mr::set_current_device_resource(&async_mr);

  int n{1 << 20};
  int block_sz = 256;
  int num_blocks{(n + block_sz - 1) / block_sz};
  std::size_t bytes{n * sizeof(int)};

  cudaStream_t stream_a{};
  cudaStream_t stream_b{};

  cudaStreamCreate(&stream_a);
  cudaStreamCreate(&stream_b);

  SECTION("Unsafe: rmm::device_buffer freed while still being written")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      rmm::device_buffer buffer(bytes, stream_a, &async_mr);
      cudaMemsetAsync(buffer.data(), 0xCC, bytes, stream_a);
      cudaStreamSynchronize(stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_b>>>(
        static_cast<int*>(buffer.data()), static_cast<int*>(output.data()), n);
    }

    // buffer is out of scope and therefore its memory could be reused on stream_a
    // meanwhile kernel may still be reading from it on stream_b...

    {
      // This exercises the use-after-free. It is not guaranteed to reproduce on all systems.
      // However, on CUDA 11.5 with a Quadro GV100 (16GB) the memory allocated overlaps foo and
      // the allocation and memset are fast enough to overlap the `cudaMemcpyAsync` on `stream_b`
      // above
      rmm::device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xff, 100 * bytes, stream_a);
    }

    cudaStreamSynchronize(stream_b);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);  // technically this could fail
  }

  SECTION("Safe: synchronize streams before and after cross-stream use.")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      rmm::device_buffer buffer(bytes, stream_a, &async_mr);
      cudaMemsetAsync(buffer.data(), 0xCC, bytes, stream_a);
      cudaStreamSynchronize(stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_b>>>(
        static_cast<int*>(buffer.data()), static_cast<int*>(output.data()), n);
      cudaStreamSynchronize(stream_b);
    }

    // buffer is out of scope, but only after the kernel finished writing to output.

    {
      // Since there is no use-after-free, this code cannot overwrite the contents of `output` as in
      // the `UseAfterFree` test.
      rmm::device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xff, 100 * bytes, stream_a);
    }

    cudaStreamSynchronize(stream_b);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);
  }

  SECTION("Safe: RAII rmm::device_buffer used on same stream as it is freed")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      rmm::device_buffer buffer(bytes, stream_a, &async_mr);
      cudaMemsetAsync(buffer.data(), 0xCC, bytes, stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        static_cast<int*>(buffer.data()), static_cast<int*>(output.data()), n);
    }

    // buffer is out of scope, but kernel and memcpy ran on the same stream so no synchronization
    // necessary

    {
      // Since there is no use-after-free, this code cannot overwrite the contents of `output` as in
      // the `UseAfterFree` test.
      rmm::device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xff, 100 * bytes, stream_a);
    }

    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);
  }

  cudaStreamDestroy(stream_a);
  cudaStreamDestroy(stream_b);

  rmm::mr::set_current_device_resource(nullptr);
}
