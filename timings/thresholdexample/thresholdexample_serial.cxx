#include "thresholdexample.h"

#include <dax/cont/DeviceAdapterSerial.h>

int main(int, char *[])
{
  return ThresholdExample<dax::cont::DeviceAdapterTagSerial>::Run("Serial STL");
}
