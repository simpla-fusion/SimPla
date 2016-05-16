# The values below are for an un-installed copy of MESHKIT used directly
# from its build build directory.  These values will be overridden below
# for installed copies of MESHKIT.
SET(MESHKIT_LIBS "-L/work/build/meshkit-1.4.1/src/.libs ")
SET(MESHKIT_INCLUDES "-I/work/build/meshkit-1.4.1/src/core
                -I/work/build/meshkit-1.4.1/src/lemon
                -I/work/build/meshkit-1.4.1/src/extern
                -I/work/build/meshkit-1.4.1/src/algs
                -I/work/build/meshkit-1.4.1/src/core
                -I/work/build/meshkit-1.4.1/src/lemon
                -I/work/build/meshkit-1.4.1/src/extern
                -I/work/build/meshkit-1.4.1/src/algs")


SET(MESHKIT_CXXFLAGS " -pipe -Wall -O2 -DNDEBUG -std=c++0x")
SET(MESHKIT_CPPFLAGS "  -DHAVE_VSNPRINTF -D_FILE_OFFSET_BITS=64 -DUSE_MPI -DHAVE_IGEOM -DHAVE_OCC -DHAVE_OCC_IGES -DHAVE_OCC_STEP -DHAVE_OCC_STL -I/pkg/sigma/1.2/include -DHAVE_IMESH  -I/usr/lib/x86_64-linux-gnu/hdf5/openmpi/include -isystem /usr/lib/x86_64-linux-gnu/hdf5/openmpi/include  -I/pkg/sigma/1.2/include  -DHAVE_FBIGEOM  -I/pkg/sigma/1.2/include  -DHAVE_IREL  -I/pkg/sigma/1.2/include  -DHAVE_MOAB -DHAVE_PARALLEL_MOAB -DHAVE_CGM")
SET(MESHKIT_CFLAGS " -pipe -Wall -O2 -DNDEBUG ")
SET(MESHKIT_FFLAGS "  -O2")
SET(MESHKIT_FCFLAGS "  -O2")
SET(MESHKIT_LDFLAGS " ")


#SET(MESHKIT_LIBRARIES "-lMeshKit -L/usr/lib   -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib -lhdf5 -lmpi -lmpi_cxx -lz -lm -L/pkg/sigma/1.2/lib -lemon -liRel -liMesh -lMOAB -lFBiGeomMOAB -lcgm -liGeom -lcgm  -lTKSTL -lTKSTEP -lTKSTEP209 -lTKSTEPAttr -lTKSTEPBase -lTKXSBase -lTKIGES -lTKXSBase -lTKBinL -lTKLCAF -lTKCDF -lTKCAF -lTKHLR -lTKOffset -lTKShHealing -lTKFillet -lTKFeat -lTKBool -lTKBO -lTKPrim -lTKMesh -lTKTopAlgo -lTKGeomAlgo -lTKBRep -lTKGeomBase -lTKG3d -lTKG2d -lTKMath -lTKernel")

SET(MESHKIT_LIBRARIES "-lMeshKit -lemon -L/usr/lib     -lmpi -lmpi_cxx  -lmetis -lstdc++   -lz   -ldl -lm     -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi/lib  -lnetcdf   -lhdf5 \
  -L/pkg/sigma/1.2/lib  -lcgm -liMesh -lMOAB -lFBiGeomMOAB -lMOAB  -liRel -liMesh    \
  -lcgm  -lTKSTL -lTKSTEP -lTKSTEP209 -lTKSTEPAttr -lTKSTEPBase -lTKXSBase -lTKIGES -lTKXSBase -lTKBinL -lTKLCAF -lTKCDF -lTKCAF \
  -lTKHLR -lTKOffset -lTKShHealing -lTKFillet -lTKFeat -lTKBool -lTKBO -lTKPrim -lTKMesh -lTKTopAlgo -lTKGeomAlgo -lTKBRep -lTKGeomBase -lTKG3d -lTKG2d -lTKMath -lTKernel" )

SET(MESHKIT_CXX "mpicxx")
SET(MESHKIT_CC  "mpicc")
SET(MESHKIT_F77  "")

SET(MESHKIT_EXTERNAL_INCLUDES "    ")

# Override MESHKIT_LIBDIR and MESHKIT_INCLUDES from above with the correct
# values for the installed MESHKIT.

SET(MESHKIT_INCLUDE_DIRS "-I/pkg/sigma/1.2/include")
