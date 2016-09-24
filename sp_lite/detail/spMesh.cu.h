//
// Created by salmon on 16-9-24.
//

#ifndef SIMPLA_SPMESH_CU_H
#define SIMPLA_SPMESH_CU_H
extern "C"
{
#include "../spMesh.h"
#include "sp_device.h"
}

typedef struct
{
    uint3 dims;
    uint3 center_min;
    uint3 center_max;
    uint3 strides;
    uint3 g_strides;

    size_type num_of_cell;
    Real3 inv_dx, dx, x0;


} _spMeshDevice;

extern   __constant__ _spMeshDevice _sp_mesh;

//#define SPMeshHash(_X_, _Y_, _Z_)                                                         \
//     (__umul24((uint) ((_X_) + _sp_mesh.dims.x) % _sp_mesh.dims.x, _sp_mesh.strides.x) +    \
//      __umul24((uint) ((_Y_) + _sp_mesh.dims.y) % _sp_mesh.dims.y, _sp_mesh.strides.y) +    \
//      __umul24((uint) ((_Z_) + _sp_mesh.dims.z) % _sp_mesh.dims.z, _sp_mesh.strides.z))

INLINE __device__ int SPMeshHash(int _X_, int _Y_, int _Z_)
{
    return (__umul24((uint) ((_X_) + _sp_mesh.dims.x) % _sp_mesh.dims.x, _sp_mesh.strides.x) +
            __umul24((uint) ((_Y_) + _sp_mesh.dims.y) % _sp_mesh.dims.y, _sp_mesh.strides.y) +
            __umul24((uint) ((_Z_) + _sp_mesh.dims.z) % _sp_mesh.dims.z, _sp_mesh.strides.z));
}

INLINE __device__  int SPMeshInCenter(int x, int y, int z)
{
    return (_sp_mesh.center_min.x <= x && x < _sp_mesh.center_max.x &&
            _sp_mesh.center_min.y <= y && y < _sp_mesh.center_max.y &&
            _sp_mesh.center_min.z <= z && z < _sp_mesh.center_max.z);
}

INLINE __device__  int SPMeshInBox(int x, int y, int z)
{
    return (x >= 0 && x < _sp_mesh.dims.x &&
            y >= 0 && y < _sp_mesh.dims.y &&
            z >= 0 && z < _sp_mesh.dims.z);
}


INLINE __device__ void SPMeshPoint(int x, int y, int z, Real *rx, Real *ry, Real *rz)
{
    *rx = _sp_mesh.x0.x + (*rx + x) * _sp_mesh.dx.x;
    *ry = _sp_mesh.x0.y + (*ry + y) * _sp_mesh.dx.y;
    *rz = _sp_mesh.x0.z + (*rz + z) * _sp_mesh.dx.z;

};

#endif //SIMPLA_SPMESH_CU_H
