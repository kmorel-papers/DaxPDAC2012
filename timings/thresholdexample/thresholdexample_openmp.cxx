#include "thresholdexample.h"

#include <dax/openmp/cont/DeviceAdapterOpenMP.h>

int main(int, char *[])
{
  return ThresholdExample<dax::openmp::cont::DeviceAdapterTagOpenMP>::Run();
}
