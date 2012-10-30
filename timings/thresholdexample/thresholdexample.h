#include <stdlib.h>
#include <stdio.h>

#include <fstream>
#include <iostream>

#include <vector>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/ScheduleGenerateTopology.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>

#include <dax/worklet/Magnitude.worklet>
#include <dax/worklet/Threshold.worklet>

#include <dax/worklet/Threshold.worklet>

#include <boost/timer/timer.hpp>

template<class DeviceAdapter>
class ThresholdExample
{
private:

  static const int GRID_SIZE = 432;
  static const int NUM_TRIALS = 1;

  typedef dax::cont::ArrayContainerControlTagBasic Container;

  typedef dax::cont::UniformGrid<DeviceAdapter> UniformGridType;
  typedef dax::cont::UnstructuredGrid<
      dax::exec::CellHexahedron,Container,Container,DeviceAdapter>
      UnstructuredGridType;

  typedef dax::cont::ArrayHandle<
      dax::Scalar, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleScalar;
  typedef dax::cont::ArrayHandle<
      dax::Vector3, dax::cont::ArrayContainerControlTagBasic, DeviceAdapter>
      ArrayHandleVector;

  static UniformGridType CreateUniformGrid()
  {
    UniformGridType grid;
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

  static void WriteGradientMagnitude(const ArrayHandleScalar &data)
  {
    FILE *fd = fopen("/Users/kmorel/Downloads/gradients.dat", "wb");
    assert(fd != NULL);

    typedef typename ArrayHandleScalar::PortalConstControl PortalType;
    PortalType dataPortal = data.GetPortalConstControl();

    for (typename PortalType::IteratorType iter = dataPortal.GetIteratorBegin();
         iter != dataPortal.GetIteratorEnd();
         iter++)
      {
      dax::Scalar value = *iter;
      fwrite(&value, sizeof(dax::Scalar), 1, fd);
      }

    fclose(fd);
  }

  template<typename Stream>
  static void PrintContentsToStream(UnstructuredGridType& grid, Stream &stream)
  {
    const dax::Id num_points(grid.GetNumberOfPoints());
    const dax::Id num_cells(grid.GetNumberOfCells());

    //dump header
    stream << "# vtk DataFile Version 3.0"  << std::endl;
    stream << "vtk output" << std::endl;
    stream << "ASCII"  << std::endl;
    stream << "DATASET UNSTRUCTURED_GRID" << std::endl;
    stream << "POINTS " << num_points << " float" << std::endl;

    std::vector<dax::Vector3> contPoints(num_points);
    grid.GetPointCoordinates().CopyInto(contPoints.begin());

    for(dax::Id i=0; i < num_points; ++i)
      {
      dax::Vector3 coord = contPoints[i];
      stream << coord[0] << " " << coord[1] << " " << coord[2] << " ";
      if(i%4==3)
        {
        stream << std::endl; //pump new line after each 4th vector
        }
      }
    if(num_points%4==3)
      {
      stream << std::endl;
      }

    //print cells
    stream << "CELLS " << num_cells << " " << num_cells  * (dax::exec::CellHexahedron::NUM_POINTS+1) << std::endl;

    std::vector<dax::Id> contTopo(num_cells*dax::exec::CellHexahedron::NUM_POINTS);
    grid.GetCellConnections().CopyInto(contTopo.begin());

    dax::Id index=0;
    for(dax::Id i=0; i < num_cells; ++i,index+=8)
      {
      stream << dax::exec::CellHexahedron::NUM_POINTS << " ";
      stream << contTopo[index+0] << " ";
      stream << contTopo[index+1] << " ";
      stream << contTopo[index+2] << " ";
      stream << contTopo[index+3] << " ";
      stream << contTopo[index+4] << " ";
      stream << contTopo[index+5] << " ";
      stream << contTopo[index+6] << " ";
      stream << contTopo[index+7] << " ";

      stream << std::endl;
      }
    stream << std::endl;
    stream << "CELL_TYPES " << num_cells << std::endl;
    for(dax::Id i=0; i < num_cells; ++i)
      {
      stream << "12" << std::endl; //11 is voxel && 12 is hexa
      }
  }

public:
  static int Run(const char *description)
  {
    try
    {
    std::cout << "Reading data..." << std::endl;
    std::vector<dax::Scalar> buffer;
    ReadSupernovaData(buffer);
    std::cout << "Data read." << std::endl;

    UniformGridType grid = CreateUniformGrid();

    for (int trial = 0; trial < NUM_TRIALS; trial++)
      {
      ArrayHandleScalar inArray =
          dax::cont::make_ArrayHandle(buffer, Container(), DeviceAdapter());
      assert(grid.GetNumberOfPoints() == inArray.GetNumberOfValues());

//      std::cout << "Computing gradient..." << std::endl;
//      dax::cont::ArrayHandle<dax::Vector3> gradient;
//      dax::cont::worklet::CellGradient(
//            grid, grid.GetPointCoordinates(), inArray, gradient);
//      inArray.ReleaseResources();

//      std::cout << "Computing magnitude..." << std::endl;
//      dax::cont::ArrayHandle<dax::Scalar> magnitude;
//      dax::cont::worklet::Magnitude(gradient, magnitude);
//      gradient.ReleaseResources();

//      std::cout << "Writing gradient magnitude..." << std::endl;
//      WriteGradientMagnitude(magnitude);
//      std::cout << "Data written." << std::endl;

      std::cout << "Computing Threshold..." << std::endl;
      boost::timer::cpu_timer timer;
      timer.start();
      typedef dax::cont::ScheduleGenerateTopology<
          dax::worklet::ThresholdTopology,DeviceAdapter>
          ScheduleGenerateTopologyType;
      typedef typename ScheduleGenerateTopologyType::ClassifyResultType
          ClassifyResultType;

      dax::cont::Scheduler<DeviceAdapter> scheduler;

      ClassifyResultType classificationArray;
      scheduler.Invoke(dax::worklet::ThresholdClassify<dax::Scalar>(0.07, 1.0),
                       grid,
                       inArray,
                       classificationArray);

      ScheduleGenerateTopologyType resolveTopology(classificationArray);
      UnstructuredGridType outGrid;
      scheduler.Invoke(resolveTopology, grid, outGrid);

      // Copy grid information to host, if necessary.
      outGrid.GetCellConnections().GetPortalConstControl();
      outGrid.GetPointCoordinates().GetPortalConstControl();

      double time = (timer.elapsed().wall)/1.0e9;
      std::cout << "Time: " << time << " seconds" << std::endl;
      std::cout << "CSV," << description << "," << time << "," << trial << std::endl;
      }

//    std::cout << "Writing threshold geometry..." << std::endl;
//    std::ofstream file;
//    file.open("/Users/kmorel/Downloads/thresholdGeometry.vtk");
//    PrintContentsToStream(outGrid, file);
//    file.close();
//    std::cout << "Done." << std::endl;
    }
    catch (dax::cont::Error error)
    {
      std::cout << "Caught Dax error: " << std::endl
                << error.GetMessage() << std::endl;
      return 1;
    }

    return 0;
  }
};
