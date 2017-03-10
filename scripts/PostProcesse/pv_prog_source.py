from vtk . numpy_interface import dataset_adapter as dsa
from vtk . numpy_interface import algorithms as algs
import h5py as h5
f1=h5.File("/pkg/etc/clion/system/cmake/generated/2774870a/2774870a/Debug/example/em/tokamak0007.h5")
x=f1["/record/0/H"][:]['p']['v']
coords=algs.make_vector(x[:,0],x[:,1],x[:,2])
pts=vtk.vtkPoints()
pts.SetData(dsa.numpyTovtkDataArray ( coords , " Points "))
output.SetPoints(pts)
