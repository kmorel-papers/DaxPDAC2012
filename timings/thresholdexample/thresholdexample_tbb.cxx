#include "thresholdexample.h"

#include <dax/tbb/cont/DeviceAdapterTBB.h>

int main(int, char *[])
{
  return ThresholdExample<dax::tbb::cont::DeviceAdapterTagTBB>::Run("TBB");
}
