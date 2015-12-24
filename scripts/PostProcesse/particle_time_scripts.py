########################################################################################################
## Script
########################################################################################################
def GetUpdateTimestep(algorithm):
      """Returns the requested time value, or None if not present"""
      executive = algorithm.GetExecutive()
      outInfo = executive.GetOutputInformation(0)
      if not outInfo.Has(executive.UPDATE_TIME_STEP()):
          return None
      return outInfo.Get(executive.UPDATE_TIME_STEP())
  # This is the requested time-step. This may not be exactly equal to the
  # timesteps published in RequestInformation(). Your code must handle that
  # correctly
req_time = GetUpdateTimestep(self)

output = self.GetOutput()

  # TODO: Generate the data as you want.
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
import h5py as h5
f1=h5.File("/pkg/clion/etc/clion/system/cmake/generated/2774870a/2774870a/Debug/example/em/tokamak.h5")
x=f1["/record/H"][:,:][:,req_time,0]
y=f1["/record/H"][:,:][:,req_time,1]
z=f1["/record/H"][:,:][:,req_time,2]
coords=algs.make_vector(x,y,z)
pts=vtk.vtkPoints()
pts.SetData(dsa.numpyTovtkDataArray ( coords , "Points"))
output.SetPoints(pts)
  # Now mark the timestep produced.
output.GetInformation().Set(output.DATA_TIME_STEP(), req_time)etInformation().Set(output.DATA_TIME_STEP(), req_time)


########################################################################################################
## Script (RequestInformation)
########################################################################################################

def SetOutputTimesteps(algorithm, timesteps):
      executive = algorithm.GetExecutive()
      outInfo = executive.GetOutputInformation(0)
      outInfo.Remove(executive.TIME_STEPS())
      for timestep in timesteps:
        outInfo.Append(executive.TIME_STEPS(), timestep)
      outInfo.Remove(executive.TIME_RANGE())
      outInfo.Append(executive.TIME_RANGE(), timesteps[0])
      outInfo.Append(executive.TIME_RANGE(), timesteps[-1])
SetOutputTimesteps(self, (0, 1 , 2 , 3,4,5,6,7,8,9 ))