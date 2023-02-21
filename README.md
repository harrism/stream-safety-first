# Stream Safety First!

This is a repo of simple examples from Mark Harris's presentation for the 2023 GPU Technology
Conference, entitled: "Robust and Efficient CUDA C++ Concurrency with Stream-Ordered Allocation",
or "Stream Safety First!".

The examples are provided in the form of Catch2 unit tests where each test has one or more sections
demonstrating some unsafe concurrent CUDA C++ code pattern, and then one or more sections
demonstrate a safe alternative.

## Prerequisites

 - [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Developed on versions 11.5 and
   11.8. A version at least as new is recommended.
 - CMake 3.20.1 or newer

Only tested on Linux (Ubuntu 20.04).

## Building

The CMake configuration uses the CMake Package Manager, [CPM](https://github.com/cpm-cmake/CPM.cmake)
to fetch all other dependencies ([Catch2](https://github.com/catchorg/Catch2) and
[RAPIDS Memory Manager](https://github.com/rapidsai/rmm/)). To build, simply run cmake in an
out-of-source build directory, and then run make.

Example:

```
cd stream-safety-first
mkdir build && cd build
cmake ..
make
```

To use Ninja instead of Make:

```
cd stream-safety-first
mkdir build && cd build
cmake .. -GNinja
ninja
```

## Running

Running the built executable `stream-safety-tests` will run all unit tests. Note that multiple tests
exercise undefined behavior so it is possible that they may fail.

```
./stream-safety-tests
===============================================================================
All tests passed (17 assertions in 7 test cases)
```
