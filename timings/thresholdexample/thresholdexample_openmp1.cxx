#include "thresholdexample.h"

#include <dax/openmp/cont/DeviceAdapterOpenMP.h>

#include <omp.h>

int main(int, char *[])
{
  omp_set_num_threads(1);
  return ThresholdExample<dax::openmp::cont::DeviceAdapterTagOpenMP>::Run("OpenMP 1 Core");
}
