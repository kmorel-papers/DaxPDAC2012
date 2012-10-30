#include "thresholdexample.h"

#include <dax/tbb/cont/DeviceAdapterTBB.h>

#include <tbb/task_scheduler_init.h>

int main(int, char *[])
{
  tbb::task_scheduler_init schedulerInit(1);
  (void)schedulerInit; // Shut up compiler
  return ThresholdExample<dax::tbb::cont::DeviceAdapterTagTBB>::Run("TBB 1 core");
}
