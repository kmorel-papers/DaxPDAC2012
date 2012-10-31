
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_CUDA
#define BOOST_SP_DISABLE_THREADS

#include "pistoncompare.h"

int main(int, char *[])
{
  return RunPistonCompare("CUDA");
}
