cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

project(StreamSafetyFirst
        VERSION 00.01.00
        LANGUAGES CXX CUDA)

# ---- Dependencies ----

# ---- Get CPM ----

set(CPM_DOWNLOAD_VERSION 0.34.0)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:catchorg/Catch2@2.13.8")
CPMAddPackage("gh:rapidsai/rmm@22.12.00")

if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures" FORCE)
endif()
message ("CUDA ARCHS: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- Create binary ----

add_executable(stream-safety-tests examples.cu rmm_examples.cu)
target_link_libraries(stream-safety-tests Catch2 rmm::rmm)
target_compile_features(stream-safety-tests PRIVATE cxx_std_17)

# ---- Enable testing ----

enable_testing()
add_test(stream-safety-tests stream-safety-tests)
