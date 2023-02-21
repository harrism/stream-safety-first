/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ex  ess or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <catch2/catch.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/thrust_allocator_adaptor.hpp>

#include <thrust/host_vector.h>

#include <memory>

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

TEST_CASE("rmm::device_buffer use-after-free", "[example_3a]")
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

TEST_CASE("Stream-ordered device_vector use-after-free a", "[example_4]")
{
  auto async_mr = rmm::mr::cuda_async_memory_resource{15UL << 30};
  rmm::mr::set_current_device_resource(&async_mr);

  int n{1 << 20};
  int block_sz = 256;
  int num_blocks{(n + block_sz - 1) / block_sz};
  std::size_t bytes{n * sizeof(int)};

  cudaStream_t stream_a{};
  cudaStreamCreate(&stream_a);

  SECTION("Unsafe: rmm::device_vector freed while still being written")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      // rmm::device_vector uses the a custom allocator that uses the current RMM memory
      // resource. This is equivalent to the `stream_device_vector` used in the
      // "Stream Safety First" presentation
      rmm::device_vector<int> v{h_reference};
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        v.data().get(), static_cast<int*>(output.data()), n);
    }

    cudaStreamSynchronize(stream_a);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);  // technically this could fail
  }

  SECTION("Safe: rmm::device_vector memory freed on same stream")
  {
    thrust::host_vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      // Using an explicit allocator allows us to specify a stream for the allocations / frees
      rmm::mr::thrust_allocator<int> allocator{stream_a};
      rmm::device_vector<int> v{h_reference, allocator};
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        v.data().get(), static_cast<int*>(output.data()), n);
    }

    cudaStreamSynchronize(stream_a);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);
  }

  cudaStreamDestroy(stream_a);

  rmm::mr::set_current_device_resource(nullptr);
}

TEST_CASE("Host vector use-after-free a", "[example_5]")
{
  auto async_mr = rmm::mr::cuda_async_memory_resource{15UL << 30};
  rmm::mr::set_current_device_resource(&async_mr);

  int n{1 << 20};
  int block_sz = 256;
  int num_blocks{(n + block_sz - 1) / block_sz};
  std::size_t bytes{n * sizeof(int)};

  cudaStream_t stream_a{};
  cudaStreamCreate(&stream_a);

  SECTION("Unsafe: std::vector freed while still being copied")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      std::vector<int> v{h_reference};

      rmm::device_uvector<int> d_v{v.size(), stream_a};
      cudaMemcpyAsync(d_v.data(), v.data(), v.size() * sizeof(int), cudaMemcpyDefault, stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        d_v.data(), static_cast<int*>(output.data()), n);
    }

    cudaStreamSynchronize(stream_a);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);  // technically this could fail
  }

  SECTION("Safe: rmm::device_vector freed while still being written")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    {
      std::vector<int> v{h_reference};

      rmm::device_uvector<int> d_v{v.size(), stream_a};
      cudaMemcpyAsync(d_v.data(), v.data(), v.size() * sizeof(int), cudaMemcpyDefault, stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        d_v.data(), static_cast<int*>(output.data()), n);
      cudaStreamSynchronize(stream_a);
    }

    cudaStreamSynchronize(stream_a);
    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), output.data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);
  }

  cudaStreamDestroy(stream_a);

  rmm::mr::set_current_device_resource(nullptr);
}

class widget {
 public:
  widget(rmm::device_vector<int> const& v, cudaStream_t stream) : _v(v.size(), stream)
  {
    cudaMemcpyAsync(_v.data(), v.data().get(), v.size() * sizeof(int), cudaMemcpyDefault, stream);
  }

  widget(rmm::device_uvector<int> const& v, cudaStream_t stream) : _v(v.size(), stream)
  {
    cudaMemcpyAsync(_v.data(), v.data(), v.size() * sizeof(int), cudaMemcpyDefault, stream);
  }

  int* data() { return _v.data(); }

 private:
  rmm::device_uvector<int> _v;
};

// Create a widget from a host vector
std::unique_ptr<widget> make_widget_unsafe_sync(std::vector<int> const& input, cudaStream_t stream)
{
  rmm::device_vector<int> d_temp{input};
  return std::make_unique<widget>(d_temp, stream);
}

std::unique_ptr<widget> make_widget_unsafe_async(std::vector<int> const& input, cudaStream_t stream)
{
  rmm::device_uvector<int> d_temp{input.size(), stream};
  cudaMemcpyAsync(
    d_temp.data(), input.data(), input.size() * sizeof(int), cudaMemcpyDefault, stream);
  return std::make_unique<widget>(d_temp, stream);
}

std::unique_ptr<widget> make_widget_safe_sync(std::vector<int> const& input, cudaStream_t stream)
{
  rmm::device_uvector<int> d_temp{input.size(), stream};
  cudaMemcpyAsync(
    d_temp.data(), input.data(), input.size() * sizeof(int), cudaMemcpyDefault, stream);
  auto w = std::make_unique<widget>(d_temp, stream);
  cudaStreamSynchronize(stream);
  return w;
}

TEST_CASE("Unsafe, asynchronous and slow", "[example_6]")
{
  auto async_mr = rmm::mr::cuda_async_memory_resource{15UL << 30};
  rmm::mr::set_current_device_resource(&async_mr);

  int n{1 << 20};
  std::size_t bytes{n * sizeof(int)};

  cudaStream_t stream_a{};
  cudaStreamCreate(&stream_a);

  SECTION("Unsafe: make_widget's device_vector freed while still being copied")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    auto w = make_widget_unsafe_sync(h_reference, stream_a);

    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), w->data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);  // technically this could fail
  }

  SECTION("Unsafe: make_widget's device_uvector freed while still being copied")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    auto w = make_widget_unsafe_async(h_reference, stream_a);

    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), w->data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);  // technically this could fail
  }

  SECTION("Safe but synchronous")
  {
    std::vector<int> h_reference(n, 0xCCCCCCCC);
    rmm::device_buffer output(bytes, stream_a);

    auto w = make_widget_safe_sync(h_reference, stream_a);

    std::vector<int> h_output(n, 0);
    cudaMemcpy(h_output.data(), w->data(), bytes, cudaMemcpyDefault);
    REQUIRE(h_output == h_reference);
  }

  cudaStreamDestroy(stream_a);

  rmm::mr::set_current_device_resource(nullptr);
}
