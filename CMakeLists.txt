cmake_minimum_required(VERSION 3.17)
project(pyGGpu LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/venv/bin/python")
  set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/venv/bin/python3")
  set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python3")
endif()

find_package(Protobuf REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR "${pybind11_CMAKE_DIR}" CACHE PATH "Path to pybind11 CMake files")

pybind11_add_module(pyggpu MODULE
  c++/src/compiler.cpp
  c++/src/kernel_registry.cpp
  c++/bindings/pyggpu.cpp
  c++/serialization/ir.pb.cc
)

# On many Linux distros llvm-config is versioned (e.g. llvm-config-20),
# so auto-discover a matching LLVMConfig.cmake path if LLVM_DIR is unset.
if(NOT DEFINED LLVM_DIR)
  foreach(_llvm_ver 20 19 18 17 16 15 14)
    if(EXISTS "/usr/lib/llvm-${_llvm_ver}/lib/cmake/llvm/LLVMConfig.cmake")
      set(LLVM_DIR "/usr/lib/llvm-${_llvm_ver}/lib/cmake/llvm" CACHE PATH "LLVM CMake directory" FORCE)
      break()
    endif()
  endforeach()
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

# LLJIT requires ORC/ExecutionEngine-related components in addition to core IR support.
llvm_map_components_to_libnames(LLVM_LIBS
  Core
  Support
  OrcJIT
  ExecutionEngine
  RuntimeDyld
  nativecodegen
  native
  NVPTX
)

target_link_directories(pyggpu PRIVATE
  /usr/local/cuda/lib64
  /usr/local/cuda/lib64/stubs
  /usr/lib/x86_64-linux-gnu
  /usr/lib/x86_64-linux-gnu/stubs
  /usr/lib/wsl/lib
)

target_link_libraries(pyggpu PRIVATE 
  ${LLVM_LIBS}
  cuda
  protobuf::libprotobuf
)

target_include_directories(pyggpu PRIVATE 
  c++/src
  ${LLVM_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  /usr/local/cuda/include
)

target_compile_definitions(pyggpu PRIVATE 
  VERSION_INFO=${PROJECT_VERSION}
  ${LLVM_DEFINITIONS}
)

set_target_properties(pyggpu PROPERTIES
PREFIX ""
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS pyggpu LIBRARY DESTINATION .)
