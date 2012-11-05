
// Sets thrust backend if using a thrust device adapter
#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/ScheduleGenerateTopology.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/worklet/Threshold.worklet>

#define SPACE thrust::detail::default_device_space_tag

#include <piston/image3d.h>
#include <piston/threshold_geometry.h>


#include <boost/timer/timer.hpp>

static const int GRID_SIZE = 432;
static const int NUM_TRIALS = 20;

static dax::cont::UniformGrid<> CreateUniformGrid()
{
  dax::cont::UniformGrid<> grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(GRID_SIZE-1, GRID_SIZE-1, GRID_SIZE-1));
  return grid;
}

static void ReadSupernovaData(std::vector<dax::Scalar> &buffer)
{
  assert(sizeof(float) == sizeof(dax::Scalar));

  FILE *fd = fopen("/Users/kmorel/data/supernova/normal_1349.dat", "rb");
  assert(fd != NULL);

  buffer.resize(GRID_SIZE*GRID_SIZE*GRID_SIZE);
  fread(&buffer.front(), sizeof(float), GRID_SIZE*GRID_SIZE*GRID_SIZE, fd);
  assert(ferror(fd) == 0);

  fclose(fd);
}

// Really? PISTON doesn't have a way to attach a field to an image without
// pulling in VTK or defining your own class? Oh well. This should be simple to
// implement.

struct piston_scalar_image3d : piston::image3d<dax::Id, dax::Scalar, SPACE>
{
  typedef thrust::device_vector<dax::Scalar> PointDataContainer;
  PointDataContainer point_data_vector;
  typedef PointDataContainer::iterator PointDataIterator;

  piston_scalar_image3d(dax::Id xsize, dax::Id ysize, dax::Id zsize,
                        const std::vector<dax::Scalar> &data)
    : piston::image3d<dax::Id, dax::Scalar, SPACE>(xsize, ysize, zsize),
      point_data_vector(data)
  {
    assert(this->NPoints == this->point_data_vector.size());
  }

  PointDataIterator point_data_begin() {
    return this->point_data_vector.begin();
  }
  PointDataIterator point_data_end() {
    return this->point_data_vector.end();
  }
};

static void RunDax(dax::cont::UniformGrid<> &grid,
                   dax::cont::ArrayHandle<dax::Scalar> &field,
                   const char *device,
                   int trial)
{
  // Make sure data is in execution environment before starting timer.
  field.PrepareForInput();

#ifdef DAX_CUDA
  cudaEvent_t startEvent;
  cudaEventCreate(&startEvent);
  cudaEventRecord(startEvent);
  cudaEventSynchronize(startEvent);
  cudaEvent_t stopEvent;
  cudaEventCreate(&stopEvent);
#endif // DAX_CUDA

  boost::timer::cpu_timer timer;
  timer.start();
  typedef dax::cont::ScheduleGenerateTopology<dax::worklet::ThresholdTopology>
      ScheduleGenerateTopologyType;
  typedef ScheduleGenerateTopologyType::ClassifyResultType ClassifyResultType;

  dax::cont::Scheduler<> scheduler;

  ClassifyResultType classificationArray;
  scheduler.Invoke(dax::worklet::ThresholdClassify<dax::Scalar>(0.07, 1.0),
                   grid,
                   field,
                   classificationArray);

  ScheduleGenerateTopologyType resolveTopology(classificationArray);
  resolveTopology.SetRemoveDuplicatePoints(false);
  dax::cont::UnstructuredGrid<dax::exec::CellHexahedron> outGrid;
  scheduler.Invoke(resolveTopology, grid, outGrid);

#ifdef DAX_CUDA
  // I belive all computation will be synchronized, but to be sure synchronize
  // an event.
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
#endif //DAX_CUDA

  double time = (timer.elapsed().wall)/1.0e9;
  std::cout << "Dax," << device << "," << time << "," << trial << std::endl;

//  std::cout << "Num output cells: " << outGrid.GetNumberOfCells() << std::endl;
}

static void RunPiston(piston_scalar_image3d &image,
                      const char *device,
                      int trial)
{
  piston::threshold_geometry<piston_scalar_image3d> threshold(image, 0.07, 1.0);

#ifdef DAX_CUDA
  cudaEvent_t startEvent;
  cudaEventCreate(&startEvent);
  cudaEventRecord(startEvent);
  cudaEventSynchronize(startEvent);
  cudaEvent_t stopEvent;
  cudaEventCreate(&stopEvent);
#endif // DAX_CUDA

  boost::timer::cpu_timer timer;
  timer.start();

  threshold();

#ifdef DAX_CUDA
  // Synchronize all CUDA operations just in case
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
#endif //DAX_CUDA

  double time = (timer.elapsed().wall)/1.0e9;
  std::cout << "PISTON," << device << "," << time << "," << trial << std::endl;

//  std::cout << "Num output cells: " << threshold.valid_cell_indices.size() << std::endl;
}

int RunPistonCompare(const char *device)
{
  try
    {
    std::vector<dax::Scalar> buffer;
    ReadSupernovaData(buffer);

    {
    dax::cont::ArrayHandle<dax::Scalar> inArray =
        dax::cont::make_ArrayHandle(buffer);

    dax::cont::UniformGrid<> grid = CreateUniformGrid();
    assert(grid.GetNumberOfPoints() == inArray.GetNumberOfValues());

    for (int trial = 0; trial < NUM_TRIALS; trial++)
      {
      RunDax(grid, inArray, device, trial);
      }
    }

    {
    piston_scalar_image3d piston_image(GRID_SIZE, GRID_SIZE, GRID_SIZE, buffer);

    for (int trial = 0; trial < NUM_TRIALS; trial++)
      {
      RunPiston(piston_image, device, trial);
      }
    }
    }
  catch (dax::cont::Error &error)
    {
    std::cout << "Caught Dax error: " << std::endl
              << error.GetMessage() << std::endl;
    return 1;
    }
  catch (std::exception &error)
    {
    std::cout << "Caught standard exception: " << error.what() << std::endl;
    }
  catch (...)
    {
    std::cout << "Caught unknown error" << std::endl;
    }

  return 0;
}
