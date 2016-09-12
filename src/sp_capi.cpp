/*
 * spIO.cpp
 *
 *  Created on: 2016年6月18日
 *      Author: salmon
 */

#include <memory>
#include <cassert>
#include "gtl/logo.h"
#include "io/IO.h"
#include "parallel/MPIComm.h"
#include "data_model/DataSet.h"
#include "data_model/DataType.h"
#include "data_model/DataSpace.h"
#include "parallel/MPIDataType.h"
#include "numeric/rectangle_distribution.h"
#include "numeric/multi_normal_distribution.h"

extern "C"
{
#include "../sp_lite/spMPI.h"
#include "../sp_lite/spDataType.h"
#include "../sp_lite/spIOStream.h"
#include "../sp_lite/spRandom.h"
};
using namespace simpla;

void ShowSimPlaLogo() { MESSAGE << ShowLogo() << std::endl; }

struct spDataType_s
{
    simpla::data_model::DataType self;

    simpla::MPIDataType m_mpi_type_;

};

int spDataTypeCreate(spDataType **dtype, int type_tag, int s)
{
    if (dtype != nullptr)
    {
        *dtype = new spDataType;

        switch (type_tag)
        {
            case SP_TYPE_float:
                (*dtype)->self = simpla::data_model::DataType::create<float>();
                break;
            case SP_TYPE_double:
                (*dtype)->self = simpla::data_model::DataType::create<double>();
                break;

            case SP_TYPE_int:
                (*dtype)->self = simpla::data_model::DataType::create<int>();
                break;
            case SP_TYPE_uint:
                (*dtype)->self = simpla::data_model::DataType::create<unsigned int>();
                break;
            case SP_TYPE_long:
                (*dtype)->self = simpla::data_model::DataType::create<long>();
                break;
            case SP_TYPE_int64_t:
                (*dtype)->self = simpla::data_model::DataType::create<int64_t>();
                break;
            default:
                (*dtype)->self.size_in_byte(s);
                break;
        }
    }
    return SP_SUCCESS;
}

int spDataTypeDestroy(spDataType **dtype)
{
    if (dtype != nullptr && *dtype != nullptr)
    {
        delete *dtype;
        *dtype = nullptr;
    }
    return SP_SUCCESS;
}

int spDataTypeCopy(spDataType *first, spDataType const *second)
{
    first->self = second->self;
    first->m_mpi_type_ = second->m_mpi_type_;
};

int spDataTypeSizeInByte(spDataType const *dtype) { return (int) dtype->self.size_in_byte(); }

int spDataTypeIsValid(spDataType const *dtype) { return dtype->self.is_valid(); }

int spDataTypeExtent(spDataType *dtype, int rank, const int *d)
{
    size_type t_d[rank];
    for (int i = 0; i < rank; ++i) { t_d[i] = (size_type) (d[i]); }
    dtype->self.extent(rank, t_d);
    return SP_SUCCESS;
}
int spDataTypeAdd(spDataType *dtype, int offset, char const *name, spDataType const *other)
{
    dtype->self.push_back(other->self, name, (size_type) offset);

    return SP_SUCCESS;
}

int spDataTypeAddArray(spDataType *dtype,
                       size_type offset,
                       char const *name,
                       int type_tag,
                       size_type n,
                       size_type const *dims)
{
//    spDataType *ele;
//
//    spDataTypeCreate(&ele, type_tag, 0);
//
//    if (dims == nullptr) { spDataTypeExtent(ele, 1, &n); } else { spDataTypeExtent(ele, n, dims); }
//
//    spDataTypeAdd(dtype, offset, name, ele);
//
//    spDataTypeDestroy(&ele);

    return SP_DO_NOTHING;
}

MPI_Datatype spDataTypeMPIType(struct spDataType_s const *dtype)
{
    MPI_Datatype res = MPI_DATATYPE_NULL;

    if (spMPIComm() != MPI_COMM_NULL) { MPI_Type_dup(simpla::MPIDataType::create((dtype)->self).type(), &res); }

    return (res);
};

struct spIOStream_s { std::shared_ptr<simpla::io::IOStream> self; };

typedef struct spIOStream_s spIOStream;

int spIOStreamCreate(spIOStream **os)
{
    *os = new spIOStream;
    (*os)->self = std::make_shared<simpla::io::HDF5Stream>();
    return SP_SUCCESS;
}

int spIOStreamDestroy(spIOStream **os)
{
    if (*os != nullptr) { delete *os; };
    *os = nullptr;
    return SP_SUCCESS;

};

int spIOStreamPWD(spIOStream *os, char *url)
{
    strcpy(url, os->self->pwd().c_str());
    return SP_SUCCESS;

};

int spIOStreamOpen(spIOStream *os, const char *url)
{
    assert(os->self != nullptr);
    os->self->open(url);
    return SP_SUCCESS;

}

int spIOStreamClose(spIOStream *os)
{
    os->self->close();
    return SP_SUCCESS;
}

int spIOStreamWriteSimple(spIOStream *os,
                          const char *url,
                          struct spDataType_s const *d_type,
                          void *d,
                          int ndims,
                          int const *dims,
                          int const *start,
                          int const *stride,
                          int const *count,
                          int const *block,
                          int const *g_dims,
                          int const *g_start,
                          int flag)
{

    simpla::data_model::DataSet dset;

    size_type l_dims[10];
    size_type l_start[10];
    size_type l_stride[10];
    size_type l_count[10];
    size_type l_block[10];
    size_type l_g_dims[10];
    size_type l_g_start[10];


    for (int i = 0; i < ndims; ++i)
    {
        l_dims[i] = (dims != NULL) ? (size_type) dims[i] : 1;
        l_start[i] = (start != NULL) ? (size_type) start[i] : 0;
        l_stride[i] = (stride != NULL) ? (size_type) stride[i] : 1;
        l_count[i] = (count != NULL) ? (size_type) count[i] : 1;
        l_block[i] = (block != NULL) ? (size_type) block[i] : 1;
        l_g_dims[i] = (g_dims != NULL) ? (size_type) g_dims[i] : l_dims[i];
        l_g_start[i] = (g_start != NULL) ? (size_type) g_start[i] : l_start[i];

    }

    dset.data_type = d_type->self;
    dset.data_space = simpla::data_model::DataSpace(ndims, l_g_dims);
    dset.data_space.select_hyperslab(l_g_start, l_stride, l_count, l_block);
    dset.memory_space = simpla::data_model::DataSpace(ndims, l_dims);
    dset.memory_space.select_hyperslab(l_start, l_stride, l_count, l_block);

    dset.data = std::shared_ptr<void>(d, simpla::tags::do_nothing());

    VERBOSE << os->self->write(url, dset, flag) << std::endl;
    return SP_DO_NOTHING;

}

int spRandomMultiNormalDistributionInCell(int const *min,
                                          int const *max,
                                          int const *strides,
                                          unsigned int pic,
                                          Real *rx,
                                          Real *ry,
                                          Real *rz,
                                          Real *vx,
                                          Real *vy,
                                          Real *vz)
{
    std::mt19937 rnd_gen(6);

    size_t number = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]) * pic;


    rnd_gen.discard(number);

    rectangle_distribution<3> x_dist;

    multi_normal_distribution<3> v_dist;


    for (int i = min[0]; i < max[0]; ++i)
        for (int j = min[1]; j < max[1]; ++j)
            for (int k = min[2]; k < max[2]; ++k)
            {
                int s = (i * strides[0] + j * strides[1] + k * strides[2]);
                for (int l = 0; l < pic; ++l)
                {
                    Real x[3], v[3];
                    x_dist(rnd_gen, &x[0]);
                    v_dist(rnd_gen, &v[0]);

                    rx[s + l] = x[0];
                    ry[s + l] = x[1];
                    rz[s + l] = x[2];

                    vx[s + l] = v[0];
                    vy[s + l] = v[1];
                    vz[s + l] = v[2];
                }

            }

}
//
//
//int spMPITopologyNumOfDims() { return GLOBAL_COMM.topology_num_of_dims(); }
//
//int const *spMPITopologyDims() { return GLOBAL_COMM.topology_dims(); }
//
//int spMPITopologyNumOfNeighbours() { return GLOBAL_COMM.topology_num_of_neighbours(); }
//
//int const *spMPITopologyNeighbours() { return GLOBAL_COMM.topology_neighbours(); }
//
//void spMPITopologyCoordinate(int rank, int *d) { GLOBAL_COMM.topology_coordinate(rank, d); }
//
//int spMPITopologyRank(int const *d) { return GLOBAL_COMM.rank(d); };


