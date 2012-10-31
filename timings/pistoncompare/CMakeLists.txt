cmake_minimum_required(VERSION 2.8)

project(pistoncompare CXX)

find_package(Dax REQUIRED)

find_package(Boost REQUIRED COMPONENTS timer system)
link_libraries(${Boost_LIBRARIES})

find_path(PISTON_INCLUDE
  NAMES piston/piston_math.h
  DOC "PISTON headers"
  )

DaxConfigureOpenMP(REQUIRED)
DaxConfigureCuda(REQUIRED)

include_directories(${PISTON_INCLUDE})

add_executable(pistoncompare_openmp
  pistoncompare.h
  pistoncompare_openmp.cxx
  )
set_source_files_properties(pistoncompare.h
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

cuda_add_executable(pistoncompare_cuda
  pistoncompare.h
  pistoncompare_cuda.cu
  )
