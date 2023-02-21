/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#define CATCH_CONFIG_MAIN

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <catch2/catch.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

namespace {

__global__ void kernel(int* input, int* output, int n, int iterations = 1000)
{
  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    for (int i = 0; i < iterations; i++) {
      output[idx] = input[idx] * input[idx] + i;
    }
  }
}

class device_buffer {
 public:
  device_buffer(std::size_t size, cudaStream_t stream) : _size(size), _stream(stream)
  {
    cudaMallocAsync(&_data, _size, _stream);
  }

  ~device_buffer() { cudaFreeAsync(_data, _stream); }

  void* data() { return _data; }

 private:
  void* _data{};
  std::size_t _size;
  cudaStream_t _stream;
};

}  // namespace

class DataRaceFixture {
 protected:
  DataRaceFixture()
  {
    int device{};
    cudaGetDevice(&device);
    // Construct explicit pool
    cudaMemPoolProps pool_props{};
    pool_props.allocType     = cudaMemAllocationTypePinned;
    pool_props.handleTypes   = cudaMemHandleTypePosixFileDescriptor;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id   = device;
    cudaMemPoolCreate(&cuda_pool_handle_, &pool_props);
    std::size_t free{};
    std::size_t total{};
    cudaMemGetInfo(&free, &total);
    cudaMemPoolSetAttribute(cuda_pool_handle_, cudaMemPoolAttrReleaseThreshold, &total);
    cudaDeviceSetMemPool(device, cuda_pool_handle_);

    // initialize input to sequence [0, n)
    h_input.resize(n);
    std::iota(h_input.begin(), h_input.end(), 0);
    // page-lock the host data to enable asynchronous copies
    cudaHostRegister(h_input.data(), bytes, cudaHostRegisterDefault);

    // initialize output to zeros
    h_output.resize(n);
    // page-lock the host data to enable asynchronous copies
    cudaHostRegister(h_output.data(), bytes, cudaHostRegisterDefault);

    // generate host reference vector where each element is the square of the
    // input + 999
    h_reference.resize(n);
    std::transform(
      h_input.begin(), h_input.end(), h_reference.begin(), [](int x) { return (x * x) + 999; });

    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    cudaEventCreateWithFlags(&event_a, cudaEventDisableTiming);

    int* big;
    cudaMallocAsync(&big, 0.9 * total, stream_a);
    cudaFreeAsync(big, stream_a);
  }

  ~DataRaceFixture()
  {
    cudaStreamDestroy(stream_a);
    cudaStreamDestroy(stream_b);

    cudaHostUnregister(h_output.data());
    cudaHostUnregister(h_input.data());

    cudaEventDestroy(event_a);
  }

  int n{1 << 20};
  int block_sz = 256;
  int num_blocks{(n + block_sz - 1) / block_sz};
  std::size_t bytes{n * sizeof(int)};

  std::vector<int> h_input{};
  std::vector<int> h_output{};
  std::vector<int> h_reference{};

  int* input{};
  int* output{};
  int* foo{};
  int* bar{};

  cudaStream_t stream_a{};
  cudaStream_t stream_b{};
  cudaEvent_t event_a{};

  cudaMemPool_t cuda_pool_handle_{};
};

TEST_CASE_METHOD(DataRaceFixture, "Simple data race", "[example_1]")
{
  cudaMalloc(&input, bytes);
  cudaMalloc(&foo, bytes);
  cudaMalloc(&bar, bytes);

  cudaMemcpyAsync(input, h_input.data(), bytes, cudaMemcpyDefault, stream_a);

  SECTION("Unsafe: Data race between kernel and cudaMemcpyAsync")
  {
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, foo, n);
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, bar, n);
    // data race: possible read of `foo` on `stream_b` before `kernel` writes to it on `stream_a`
    cudaMemcpyAsync(h_output.data(), foo, bytes, cudaMemcpyDefault, stream_b);

    cudaStreamSynchronize(stream_b);
    REQUIRE(h_output != h_reference);
  }

  SECTION("Safe: No data race using a cudaStreamWaitEvent")
  {
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, foo, n);
    cudaEventRecord(event_a, stream_a);
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, bar, n);
    // prevent data race by waiting on event
    cudaStreamWaitEvent(stream_b, event_a);
    cudaMemcpyAsync(h_output.data(), foo, bytes, cudaMemcpyDefault, stream_b);

    cudaStreamSynchronize(stream_b);
    REQUIRE(h_output == h_reference);
  }

  cudaDeviceSynchronize();

  cudaFree(bar);
  cudaFree(foo);
  cudaFree(input);
}

TEST_CASE_METHOD(DataRaceFixture, "Use after free", "[example_2]")
{
  cudaMallocAsync(&input, bytes, stream_a);
  cudaMallocAsync(&foo, bytes, stream_a);
  cudaMallocAsync(&bar, bytes, stream_a);

  cudaMemcpyAsync(input, h_input.data(), bytes, cudaMemcpyDefault, stream_a);

  SECTION("Unsafe: use-after-free of foo on stream_b")
  {
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, foo, n);
    cudaEventRecord(event_a, stream_a);
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, bar, n, 1);
    // prevent data race by waiting on event
    cudaStreamWaitEvent(stream_b, event_a);
    cudaMemcpyAsync(h_output.data(), foo, bytes, cudaMemcpyDefault, stream_b);

    cudaFreeAsync(bar, stream_a);
    // use-after-free of foo on stream_b
    cudaFreeAsync(foo, stream_a);
    cudaFreeAsync(input, stream_a);

    // This exercises the use-after-free. It is not guaranteed to reproduce on all systems.
    // However, on CUDA 11.5 with a Quadro GV100 (16GB) the memory allocated overlaps foo and
    // the allocation and memset are fast enough to overlap the `cudaMemcpyAsync` on `stream_b`
    // above
    int* racer{};
    cudaMallocAsync(&racer, 100 * bytes, stream_a);
    cudaMemsetAsync(racer, 0xcc, 100 * bytes, stream_a);
    cudaFreeAsync(racer, stream_a);

    cudaStreamSynchronize(stream_b);
    REQUIRE(h_output != h_reference);
  }

  SECTION("Safe: Free foo on stream_b where last used")
  {
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, foo, n);
    cudaEventRecord(event_a, stream_a);
    kernel<<<num_blocks, block_sz, 0, stream_a>>>(input, bar, n, 1);
    // prevent data race by waiting on event
    cudaStreamWaitEvent(stream_b, event_a);
    cudaMemcpyAsync(h_output.data(), foo, bytes, cudaMemcpyDefault, stream_b);

    cudaFreeAsync(bar, stream_a);
    // No use-after-free of foo on stream_b since we free it on stream_b
    cudaFreeAsync(foo, stream_b);
    cudaFreeAsync(input, stream_a);

    // Since there is no use-after-free, this code cannot overwrite the contents of `foo` as in
    // the `UseAfterFree` test.
    int* racer{};
    cudaMallocAsync(&racer, 100 * bytes, stream_a);
    cudaMemsetAsync(racer, 0xcc, 100 * bytes, stream_a);
    cudaFreeAsync(racer, stream_a);

    cudaStreamSynchronize(stream_b);
    REQUIRE(h_output == h_reference);
  }
}

TEST_CASE_METHOD(DataRaceFixture, "device_buffer use-after-free", "[example_3]")
{
  SECTION("Unsafe: RAII device_buffer use after free")
  {
    device_buffer output(bytes, stream_a);

    {
      device_buffer input(bytes, stream_a);
      cudaMemcpyAsync(input.data(), h_input.data(), bytes, cudaMemcpyDefault, stream_a);
      cudaStreamSynchronize(stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_b>>>(
        static_cast<int*>(input.data()), static_cast<int*>(output.data()), n);
    }

    // input is out of scope and therefore its memory could be reused on stream_a
    // meanwhile kernel may still be reading from it on stream_b...

    {
      // This exercises the use-after-free. It is not guaranteed to reproduce on all systems.
      // However, on CUDA 11.5 with a Quadro GV100 (16GB) the memory allocated overlaps foo and
      // the allocation and memset are fast enough to overlap the `kernel` on `stream_b`
      // above
      device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xcc, 100 * bytes, stream_a);
    }

    cudaMemcpyAsync(h_output.data(), output.data(), bytes, cudaMemcpyDefault, stream_b);
    cudaStreamSynchronize(stream_b);
    REQUIRE(h_output == h_reference);  // Technically this could fail
  }

  SECTION("Safe: synchronize streams before and after cross-stream use.")
  {
    device_buffer output(bytes, stream_a);

    {
      device_buffer input(bytes, stream_a);
      cudaMemcpyAsync(input.data(), h_input.data(), bytes, cudaMemcpyDefault, stream_a);
      cudaStreamSynchronize(stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_b>>>(
        static_cast<int*>(input.data()), static_cast<int*>(output.data()), n);
      cudaStreamSynchronize(stream_b);
    }

    // input is out of scope, but only after the kernel finished writing to output.

    {
      // Since there is no use-after-free, this code cannot overwrite the contents of `output` as in
      // the `UseAfterFree` test.
      device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xcc, 100 * bytes, stream_a);
    }

    cudaMemcpyAsync(h_output.data(), output.data(), bytes, cudaMemcpyDefault, stream_a);
    cudaStreamSynchronize(stream_a);
    REQUIRE(h_output == h_reference);
  }

  SECTION("Safe: RAII device_buffer used on same stream as it is freed")
  {
    device_buffer output(bytes, stream_a);

    {
      device_buffer input(bytes, stream_a);
      cudaMemcpyAsync(input.data(), h_input.data(), bytes, cudaMemcpyDefault, stream_a);
      kernel<<<num_blocks, block_sz, 0, stream_a>>>(
        static_cast<int*>(input.data()), static_cast<int*>(output.data()), n);
    }

    // input is out of scope, but kernel and memcpy ran on the same stream so no synchronization
    // necessary

    {
      // Since there is no use-after-free, this code cannot overwrite the contents of `output` as in
      // the `UseAfterFree` test.
      device_buffer racer(100 * bytes, stream_a);
      cudaMemsetAsync(racer.data(), 0xcc, 100 * bytes, stream_a);
    }

    cudaMemcpyAsync(h_output.data(), output.data(), bytes, cudaMemcpyDefault, stream_a);
    cudaStreamSynchronize(stream_a);
    REQUIRE(h_output == h_reference);
  }
}
