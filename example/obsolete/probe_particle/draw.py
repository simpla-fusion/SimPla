
import h5py as H5
f1=H5.File("0016.h5")
plot(f1["/H"][:,0]["x"][:,0],f1["/H"][:,0]["x"][:,1])
 