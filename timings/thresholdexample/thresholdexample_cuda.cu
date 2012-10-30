#define BOOST_SP_DISABLE_THREADS

#include "thresholdexample.h"

#include <dax/cuda/cont/DeviceAdapterCuda.h>

int main(int, char *[])
{
  return ThresholdExample<dax::cuda::cont::DeviceAdapterTagCuda>::Run("CUDA");
}
