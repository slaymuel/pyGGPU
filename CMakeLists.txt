cmake_minimum_required(VERSION 3.15)
project(pyGGpu LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS "${CMAKE_SOURCE_DIR}/venv/bin/python")
  set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/venv/bin/python3")
  set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/venv/bin/python3")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(pybind11_DIR "${pybind11_CMAKE_DIR}" CACHE PATH "Path to pybind11 CMake files")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(pyggpu MODULE
  c++/bindings/pyggpu.cpp
  c++/serialization/ir.pb.cc
)

find_package(Protobuf REQUIRED)
target_link_libraries(pyggpu PRIVATE protobuf::libprotobuf)

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

find_package(LLVM CONFIG REQUIRED)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} in ${LLVM_DIR}")

target_include_directories(pyggpu PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(pyggpu PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_LIBS Core Support)
target_link_libraries(pyggpu PRIVATE ${LLVM_LIBS})

target_include_directories(pyggpu PRIVATE c++/src)

target_compile_definitions(pyggpu PRIVATE VERSION_INFO=${PROJECT_VERSION})

set_target_properties(pyggpu PROPERTIES
PREFIX ""
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

install(TARGETS pyggpu LIBRARY DESTINATION .)
