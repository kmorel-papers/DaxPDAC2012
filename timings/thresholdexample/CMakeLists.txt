cmake_minimum_required(VERSION 2.8)

project(thresholdexample CXX)

find_package(Dax REQUIRED)

find_package(Boost REQUIRED COMPONENTS timer system)
link_libraries(${Boost_LIBRARIES})

DaxConfigureSerial(REQUIRED)
add_executable(thresholdexample_serial
  thresholdexample.h
  thresholdexample_serial.cxx
  )
set_source_files_properties(thresholdexample.h
  PROPERTIES HEADER_FILE_ONLY TRUE
  )

DaxConfigureOpenMP(REQUIRED)
add_executable(thresholdexample_openmp
  thresholdexample.h
  thresholdexample_openmp.cxx
  )

DaxConfigureTBB(REQUIRED)
add_executable(thresholdexample_tbb
  thresholdexample.h
  thresholdexample_tbb.cxx
  )

DaxConfigureCuda(REQUIRED)
cuda_add_executable(thresholdexample_cuda
  thresholdexample.h
  thresholdexample_cuda.cu
  )

