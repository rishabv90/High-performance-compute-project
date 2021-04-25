cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MACOSX_RPATH TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

project(ece569-shadowremoval)

find_package(CUDA REQUIRED)

# For some MPs it may be necessary to specify the compute capability of your
# NVIDIA GPU. In that case, simply uncomment the following two lines that start
# with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
# hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
# or '21'; Kepler-class '30', '32', '35'; or Maxwell-class '50' or '52'
set(COMPUTE_CAPABILITY 60)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --gpu-architecture;sm_${COMPUTE_CAPABILITY}")
# list(APPEND CUDA_NVCC_FLAGS -arch=sm_35)

# Set warning levels for host compilation
if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra -std=c++11")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wno-unused-function -Wno-unused-parameter -Wextra")
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -Wno-unused-function,-Wno-unused-parameter")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -O0 -std=c++11")
endif ()

# Set the correct linker options on UNIX platforms
# -lpthread inserted by ali akoglu  for compatiblity with elgato
if (UNIX)
  set(LINK_LIBRARIES "-lpthread -lm")
  include(CheckLibraryExists)
  check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
  if (HAVE_CLOCK_GETTIME)
    set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
  else ()
    message(FATAL_ERROR "clock_gettime() not found")
  endif ()
endif ()

file(GLOB SOURCES "src/*.cu")
 
include(libwb/sources.cmake)
cuda_add_library(${WBLIB} ${LIBWB_SOURCE_FILES} STATIC)
set_property(TARGET ${WBLIB} PROPERTY CXX_STANDARD 11)

cuda_add_executable(shadow_removal ${SOURCES})
target_link_libraries(shadow_removal ${WBLIB} ${LINK_LIBRARIES})
