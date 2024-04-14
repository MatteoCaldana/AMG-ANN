cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
project(spectrum)

set(CMAKE_CXX_STANDARD "20")
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Set default build type to Release.
if (NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  add_compile_options(-O3 -DNDEBUG)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)

string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UPPERCASE)
add_definitions("-DBUILD_TYPE_${CMAKE_BUILD_TYPE_UPPERCASE}")

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Determine linear algebra backend.
set(LIN_ALG "PETSc" CACHE STRING "Use Trilinos or PETSc as deal.II linear algebra backend.")

message(STATUS)
if("${LIN_ALG}" STREQUAL "Trilinos")
  add_definitions(-DLIN_ALG_TRILINOS)
elseif("${LIN_ALG}" STREQUAL "PETSc")
  add_definitions(-DLIN_ALG_PETSC)
else()
  message(FATAL_ERROR "Please select deal.II linear algebra backend with -DLinAlg=Trilinos or -DLinAlg=PETSc.")
endif()
message(STATUS "Using ${LIN_ALG} as deal.II linear algebra backend.")

# Determine whether deal.II installation supports PETSc BDDC preconditioner.
if("${LIN_ALG}" STREQUAL "PETSc")
  set(DEAL_II_WITH_BDDC OFF CACHE BOOL "Flag for BDDC support in PETSc.")

  if(${DEAL_II_WITH_BDDC})
    add_definitions(-DDEAL_II_WITH_BDDC)
    message(STATUS "Using deal.II with BDDC support.")
  else()
    set(DEAL_II_WITH_BDDC OFF)
    message(STATUS "Using deal.II without BDDC support.")
  endif()
endif()
message(STATUS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor -Wall -Wextra")


file(GLOB_RECURSE TARGET_H "${CMAKE_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE TARGET_CPP "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB OUTER_SRCS 
  "${CMAKE_SOURCE_DIR}/../common/myutils.h" 
  "${CMAKE_SOURCE_DIR}/../common/parser.h" 
  "${CMAKE_SOURCE_DIR}/../common/redirector.h" 
  "${CMAKE_SOURCE_DIR}/../common/myutils.cpp" 
  "${CMAKE_SOURCE_DIR}/../common/parser.cpp" 
  "${CMAKE_SOURCE_DIR}/../common/redirector.cpp"
)
set(TARGET_SRC ${TARGET_CPP} ${TARGET_H} ${OUTER_SRCS})
add_executable(${CMAKE_PROJECT_NAME} ${TARGET_SRC})

deal_ii_setup_target(${CMAKE_PROJECT_NAME})