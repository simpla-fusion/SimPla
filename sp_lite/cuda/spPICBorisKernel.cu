//
// Created by salmon on 16-6-14.
//
//

extern "C" {
#include <assert.h>
#include <math.h>

#include "../sp_lite_def.h"
#include "../spMesh.h"
#include "../spParticle.h"
#include "../spField.h"
#include "../spPICBoris.h"
#include "../spPICBoris_device.h"

#include "../spRandom.h"
#include "../spPhysicalConstants.h"

#include "spParallelCUDA.h"

}

#include </usr/local/cuda/include/cuda_runtime_api.h>
#include </usr/local/cuda/include/device_launch_parameters.h>


__global__ void
spParticleInitializeBorisYeeKernel(boris_particle *sp, dim3 min, dim3 max, dim3 strides, size_type pic, Real vT,
                                   Real f0, int uniform_sample)
{
    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;


    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;
    for (int x = blockIdx.x + min.x; x < max.x; x += gridDim.x)
        for (int y = blockIdx.y + min.y; y < max.y; y += gridDim.y)
            for (int z = blockIdx.z + min.z; z < max.z; z += gridDim.z)
            {
                size_type s0 = threadId + x * strides.x + y * strides.y + z * strides.z;

                for (size_type s = 0; s < pic; s += num_of_thread)
                {
                    sp->id[s0 + s] = 0;
                    sp->f[s0 + s] = f0;
                    sp->w[s0 + s] = 1.0;
                    if (uniform_sample > 0)
                    {
                        sp->f[s0 + s] *= exp(-sp->vx[s0 + s] * sp->vx[s0 + s]
                                             - sp->vy[s0 + s] * sp->vy[s0 + s]
                                             - sp->vz[s0 + s] * sp->vz[s0 + s]
                        );
                    }

                    sp->vx[s0 + s] *= vT;
                    sp->vy[s0 + s] *= vT;
                    sp->vz[s0 + s] *= vT;

                }
            }

}

int spParticleInitializeBorisYee(spParticle *sp, Real n0, Real T0, int do_import_sample)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    SP_CALL(spParticleDeploy(sp));

    size_type max_number_of_entities = spParticleGetNumberOfEntities(sp);

    int dist_type[6] = {SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_UNIFORM, SP_RAND_NORMAL, SP_RAND_NORMAL, SP_RAND_NORMAL};

    SP_CALL(spParticleInitialize(sp, dist_type));

    Real dx[3];
    SP_CALL(spMeshGetDx(m, dx));
    Real vT = (Real) sqrt(2.0 * SI_Boltzmann_constant * T0 / spParticleGetMass(sp));
    Real f0 = n0 * dx[0] * dx[1] * dx[2] / spParticleGetPIC(sp);
    size_type x_min[3], x_max[3], strides[3];
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_CENTER, x_min, x_max, strides));
    strides[0] *= spParticleGetMaxPIC(sp);
    strides[1] *= spParticleGetMaxPIC(sp);
    strides[2] *= spParticleGetMaxPIC(sp);

    size_type blocks[3] = {16, 1, 1};
    size_type threads[3]{SP_DEFAULT_NUMBER_OF_ENTITIES_IN_PAGE, 1, 1};

    void **device_data;

    spParticleGetAllAttributeData_device(sp, &device_data);

    SP_DEVICE_CALL_KERNEL(spParticleInitializeBorisYeeKernel,
                          sizeType2Dim3(blocks), sizeType2Dim3(threads),
                          (boris_particle *) device_data,
                          sizeType2Dim3(x_min), sizeType2Dim3(x_max), sizeType2Dim3(strides),
                          spParticleGetPIC(sp), vT, f0, do_import_sample
    );

    spParallelDeviceFree((void **) &device_data);
    return SP_SUCCESS;
}

/******************************************************************************************/
#define SP_MAX_NUM_OF_PARTICLE_ATTR 32
typedef struct
{
    Real inv_dv[3];
    Real cmr_dt;
    int max_pic;
    void *data[SP_MAX_NUM_OF_PARTICLE_ATTR];
    Real *rho;
    Real *J[3];
    Real *E[3];
    Real *B[3];
    cudaArray_t aE[3];
    cudaArray_t aB[3];

    int min[3];
    int max[3];
    int strides[3];

} boris_update_param;

__constant__ boris_update_param g_boris_param;

__global__ void
spBorisYeeUpdateParticleKernel()
{
    int threadId = (threadIdx.x) +
                   (threadIdx.y) * blockDim.x +
                   (threadIdx.z) * blockDim.x * blockDim.y;


    int num_of_thread = blockDim.z * blockDim.x * blockDim.y;

    __shared__ Real __align__(64) Ex[64];
    __shared__ Real __align__(64) Ey[64];
    __shared__ Real __align__(64) Ez[64];
    __shared__ Real __align__(64) Bx[64];
    __shared__ Real __align__(64) By[64];
    __shared__ Real __align__(64) Bz[64];
    //, Jx[27], Jy[27], Jz[27], rho[27];


    boris_particle *sp = (boris_particle *) g_boris_param.data;

    for (int x = blockIdx.x + g_boris_param.min[0]; x < g_boris_param.max[0]; x += gridDim.x)
        for (int y = blockIdx.y + g_boris_param.min[1]; y < g_boris_param.max[1]; y += gridDim.y)
            for (int z = blockIdx.z + g_boris_param.min[2]; z < g_boris_param.max[2]; z += gridDim.z)
            {

                __syncthreads();
                if (threadIdx.x < 4 && threadIdx.y < 4 && threadIdx.z < 4)
                {
                    int s1 = threadIdx.x * 1 + threadIdx.y * 4 + threadIdx.z * 16;
                    int s0 = (x + threadIdx.x) * g_boris_param.strides[0] +
                             (y + threadIdx.y) * g_boris_param.strides[1] +
                             (z + threadIdx.z) * g_boris_param.strides[2];


                    Ex[s1] = g_boris_param.E[0][s0];
                    Ey[s1] = g_boris_param.E[1][s0];
                    Ez[s1] = g_boris_param.E[2][s0];

                    Bx[s1] = g_boris_param.B[0][s0];
                    By[s1] = g_boris_param.B[1][s0];
                    Bz[s1] = g_boris_param.B[2][s0];

                }

                __syncthreads();
                int s0 = (x * g_boris_param.strides[0] + y * g_boris_param.strides[1] +
                          z * g_boris_param.strides[2]) * g_boris_param.max_pic;

                for (int s = threadId, se = g_boris_param.max_pic; s < se; s += num_of_thread)
                {

                    Real rx = sp->rx[s0 + s];
                    Real ry = sp->ry[s0 + s];
                    Real rz = sp->rz[s0 + s];
                    Real vx = sp->vx[s0 + s];
                    Real vy = sp->vy[s0 + s];
                    Real vz = sp->vz[s0 + s];
                    Real f = sp->f[s0 + s];
                    Real w = sp->w[s0 + s];
//                    atomicMax(&sp->id[s0 + s], 0xFF); // TODO this should be atomic

//                    spBoris(g_boris_param.cmr_dt, real2Real3(g_boris_param.inv_dv), s_c, IX, IY, IZ,
//                            rho, Jx, Jy, Jz,
//                            Ex, Ey, Ez, Bx, By, Bz, &rx, &ry, &rz, &vx, &vy, &vz, &f, &w);

                    {
                        Real ax, ay, az;
                        Real tx, ty, tz;

                        Real tt;

                        cache_gather(&ax, Ex, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][0]]);
                        cache_gather(&ay, Ey, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][1]]);
                        cache_gather(&az, Ez, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[1/*EDGE*/][2]]);
                        cache_gather(&tx, Bx, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][0]]);
                        cache_gather(&ty, By, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][1]]);
                        cache_gather(&tz, Bz, rx, ry, rz); //, id_to_shift_[sub_index_to_id_[2/*FACE*/][2]]);

                        ax *= g_boris_param.cmr_dt;
                        ay *= g_boris_param.cmr_dt;
                        az *= g_boris_param.cmr_dt;

                        tx *= g_boris_param.cmr_dt;
                        ty *= g_boris_param.cmr_dt;
                        tz *= g_boris_param.cmr_dt;

                        rx += vx * 0.5 * g_boris_param.inv_dv[0];
                        ry += vy * 0.5 * g_boris_param.inv_dv[1];
                        rz += vz * 0.5 * g_boris_param.inv_dv[2];

                        vx += ax;
                        vy += ay;
                        vz += az;

                        Real v_x, v_y, v_z;

                        v_x = vx + (vy * tz - vz * ty);
                        v_y = vy + (vz * tx - vx * tz);
                        v_z = vz + (vx * ty - vy * tx);

                        tt = 2 / (tx * tx + ty * ty + tz * tz + 1);

                        vx += ax + (v_y * tz - v_z * ty) * tt;
                        vy += ax + (v_z * tx - v_x * tz) * tt;
                        vz += ax + (v_x * ty - v_y * tx) * tt;
                        rx += vx * 0.5 * g_boris_param.inv_dv[0];
                        ry += vy * 0.5 * g_boris_param.inv_dv[1];
                        rz += vz * 0.5 * g_boris_param.inv_dv[2];

                    }


                    sp->id[s0 + s] = 0;
                    sp->rx[s0 + s] = rx;
                    sp->ry[s0 + s] = ry;
                    sp->rz[s0 + s] = rz;
                    sp->vx[s0 + s] = vx;
                    sp->vy[s0 + s] = vy;
                    sp->vz[s0 + s] = vz;
                    sp->f[s0 + s] = f;
                    sp->w[s0 + s] = w;
                }


            }


};


__global__ void
spParticleBorisYeeGatherKernel()
{
    size_type threadId = (threadIdx.x) +
                         (threadIdx.y) * blockDim.x +
                         (threadIdx.z) * blockDim.x * blockDim.y;


    size_type num_of_thread = blockDim.z * blockDim.x * blockDim.y;

//    __shared__ Real Ex[27], Ey[27], Ez[27], Bx[27], By[27], Bz[27];
    Real Jx, Jy, Jz, rho;


    boris_particle *sp = (boris_particle *) g_boris_param.data;

    for (int x = blockIdx.x + g_boris_param.min[0]; x < g_boris_param.max[0]; x += gridDim.x)
        for (int y = blockIdx.y + g_boris_param.min[1]; y < g_boris_param.max[1]; y += gridDim.y)
            for (int z = blockIdx.z + g_boris_param.min[2]; z < g_boris_param.max[2]; z += gridDim.z)
            {
                rho = 0;
                Jx = 0;
                Jy = 0;
                Jz = 0;


                size_type s0 = threadId + (x * g_boris_param.strides[0] + y * g_boris_param.strides[1] +
                                           z * g_boris_param.strides[2]) * g_boris_param.max_pic;

                for (size_type s = 0; s < g_boris_param.max_pic; s += num_of_thread)
                {
                    if (sp->id[s0 + s] != 0) { continue; }
                    Real rx = sp->rx[s0 + s];
                    Real ry = sp->ry[s0 + s];
                    Real rz = sp->rz[s0 + s];
                    Real vx = sp->vx[s0 + s];
                    Real vy = sp->vy[s0 + s];
                    Real vz = sp->vz[s0 + s];
                    Real w = sp->w[s0 + s] * sp->f[s0 + s] * (1 - rx) * (1 - ry) * (1 - rz);
                    rho += w;
                    Jx += w * vx;
                    Jy += w * vy;
                    Jz += w * vz;
                }
                __syncthreads();
                if (threadId == 0)
                {
                    size_type s0 =
                            x * g_boris_param.strides[0] + y * g_boris_param.strides[1] + z * g_boris_param.strides[2];
                    g_boris_param.rho[s0] += rho;

                    g_boris_param.J[0][s0] += Jx;
                    g_boris_param.J[1][s0] += Jy;
                    g_boris_param.J[2][s0] += Jz;
                }
                __syncthreads();

            }
};

int spParticleUpdateBorisYee(spParticle *sp, Real dt, const spField *fE, const spField *fB, spField *fRho, spField *fJ)
{
    spMesh const *m = spMeshAttributeGetMesh((spMeshAttribute *) sp);

    boris_update_param update_param;
    update_param.max_pic = (int) spParticleGetMaxPIC(sp);
    update_param.cmr_dt = dt * spParticleGetCharge(sp) / spParticleGetMass(sp);
    size_type min[3], max[3], strides[3];
    SP_CALL(spMeshGetInvDx(spMeshAttributeGetMesh((spMeshAttribute const *) sp), update_param.inv_dv));
    SP_CALL(spMeshGetArrayShape(m, SP_DOMAIN_ALL, min, max, strides));
    for (int i = 0; i < 3; ++i)
    {
        update_param.inv_dv[i] *= dt;
        update_param.min[i] = (int) min[i];
        update_param.max[i] = (int) max[i];
        update_param.strides[i] = (int) strides[i];
    }
    size_type dims[3];
    spMeshGetDims(m, dims);                  //
    struct cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);  //
    struct cudaExtent extent = {dims[0], dims[1], dims[2]};
    for (int j = 0; j < 3; ++j)
    {
        SP_CUDA_CALL(cudaMalloc3DArray(&update_param.aE[j], &desc, extent));
        SP_CUDA_CALL(cudaMalloc3DArray(&update_param.aB[j], &desc, extent));

//        SP_CUDA_CALL(cudaMemcpyToArray(update_param.aE[j], 0, 0, (void const *) update_param.E[j],
//                                       dims[0] * dims[1] * dims[2] * sizeof(Real), cudaMemcpyDefault));
//        SP_CUDA_CALL(cudaMemcpyToArray(update_param.aB[j], 0, 0, (void const *) update_param.B[j],
//                                       dims[0] * dims[1] * dims[2] * sizeof(Real), cudaMemcpyDefault));


    }

    SP_CALL(spParticleGetAllAttributeData(sp, update_param.data));
    SP_CALL(spFieldSubArray(fRho, (void **) &update_param.rho));
    SP_CALL(spFieldSubArray(fJ, (void **) update_param.J));
    SP_CALL(spFieldSubArray((spField *) fE, (void **) update_param.E));
    SP_CALL(spFieldSubArray((spField *) fB, (void **) update_param.B));

    SP_CUDA_CALL(cudaMemcpyToSymbol(g_boris_param, &update_param, sizeof(boris_update_param)));

//    SP_CALL(spParallelMemcpyToSymbol((void *) &g_boris_param, &update_param, sizeof(boris_update_param)));


    size_type blocks[3] = {SP_DEFAULT_BLOCKS / 16, 16, 1};
    size_type threads[3]{SP_DEFAULT_THREADS / 16, 4, 4};

    SP_DEVICE_CALL_KERNEL(spBorisYeeUpdateParticleKernel, sizeType2Dim3(blocks), sizeType2Dim3(threads));

    SP_DEVICE_CALL_KERNEL(spParticleBorisYeeGatherKernel, sizeType2Dim3(blocks), sizeType2Dim3(threads));
    for (int j = 0; j < 3; ++j)
    {
        SP_CUDA_CALL(cudaFreeArray(update_param.aE[j]));
        SP_CUDA_CALL(cudaFreeArray(update_param.aB[j]));
    }
    SP_CALL(spParticleSync(sp));
    SP_CALL(spFieldSync(fJ));
    SP_CALL(spFieldSync(fRho));
    return SP_SUCCESS;
}

