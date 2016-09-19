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

extern "C"
{
#include "../sp_lite/spMPI.h"
#include "../sp_lite/spIOStream.h"
#include "../sp_lite/spDataType.h"

};

//#include "numeric/rectangle_distribution.h"
//#include "numeric/multi_normal_distribution.h"

using namespace simpla;

void ShowSimPlaLogo() { MESSAGE << ShowLogo() << std::endl; };
//struct spDataType_s
//{
//    simpla::data_model::DataType self;
//
//    simpla::MPIDataType m_mpi_type_;
//
//};
//
//typedef struct spDataType_s spDataType;
//
//int spDataTypeCreate(spDataType **, int type_tag, size_type s);
//
//int spDataTypeDestroy(spDataType **);
//
//int spDataTypeCopy(spDataType *, spDataType const *);
//
////size_type spDataTypeSizeInByte(spDataType const *dtype);
//
//void spDataTypeSetSizeInByte(spDataType *dtype, size_type s);
//
//int spDataTypeIsValid(spDataType const *);
//
//int spDataTypeExtent(spDataType *, int rank, const size_type *d);
//
//int spDataTypeAdd(spDataType *dtype, size_type offset, char const *name, spDataType const *other);
//
//int spDataTypeAddArray(spDataType *dtype,
//                       size_type offset,
//                       char const *name,
//                       int type_tag,
//                       size_type n,
//                       size_type const *dims);
//
//int spDataTypeCreate(spDataType **dtype, int type_tag, size_type s)
//{
//    if (dtype != nullptr)
//    {
//        *dtype = new spDataType;
//
//        switch (type_tag)
//        {
//            case SP_TYPE_float:
//                (*dtype)->self = simpla::data_model::DataType::create<float>();
//                break;
//            case SP_TYPE_double:
//                (*dtype)->self = simpla::data_model::DataType::create<double>();
//                break;
//
//            case SP_TYPE_int:
//                (*dtype)->self = simpla::data_model::DataType::create<int>();
//                break;
//            case SP_TYPE_uint:
//                (*dtype)->self = simpla::data_model::DataType::create<unsigned int>();
//                break;
//            case SP_TYPE_long:
//                (*dtype)->self = simpla::data_model::DataType::create<long>();
//                break;
////            case SP_TYPE_int64_t:
////                (*dtype)->self = simpla::data_model::DataType::create<int64_t>();
////                break;
//            case SP_TYPE_unsigned_long:
//                (*dtype)->self = simpla::data_model::DataType::create<unsigned long>();
//                break;
//            default:
//                (*dtype)->self.size_in_byte(s);
//                break;
//        }
//    }
//    return SP_SUCCESS;
//}
//int spDataTypeDestroy(spDataType **dtype)
//{
//    if (dtype != nullptr && *dtype != nullptr)
//    {
//        delete *dtype;
//        *dtype = nullptr;
//    }
//    return SP_SUCCESS;
//}
//
//int spDataTypeCopy(spDataType *first, spDataType const *second)
//{
//    first->self = second->self;
//    first->m_mpi_type_ = second->m_mpi_type_;
//};
//size_type spDataTypeSizeInByte(spDataType const *dtype) { return dtype->self.size_in_byte(); }
//
//int spDataTypeIsValid(spDataType const *dtype) { return dtype->self.is_valid(); }
//
//int spDataTypeExtent(spDataType *dtype, int rank, const size_type *d)
//{
//    dtype->self.extent(rank, d);
//    return SP_SUCCESS;
//}
//int spDataTypeAdd(spDataType *dtype, size_type offset, char const *name, spDataType const *other)
//{
//    dtype->self.push_back(other->self, name, offset);
//
//    return SP_SUCCESS;
//}
//
//int spDataTypeAddArray(spDataType *dtype,
//                       size_type offset,
//                       char const *name,
//                       int type_tag,
//                       size_type n,
//                       size_type const *dims)
//{
//    spDataType *ele;
//
//    spDataTypeCreate(&ele, type_tag, 0);
//
//    if (dims == nullptr) { spDataTypeExtent(ele, 1, &n); } else { spDataTypeExtent(ele, n, dims); }
//
//    spDataTypeAdd(dtype, offset, name, ele);
//
//    spDataTypeDestroy(&ele);
//
//    return SP_SUCCESS;
//}
//MPI_Datatype spDataTypeMPIType(struct spDataType_s const *dtype)
//{
//    MPI_Datatype res = MPI_DATATYPE_NULL;
//
//    if (spMPIComm() != MPI_COMM_NULL) { MPI_Type_dup(simpla::MPIDataType::create((dtype)->self).type(), &res); }
//
//    return (res);
//};
simpla::data_model::DataType spDataTypeFromTag(int type_tag, size_type s)
{

    switch (type_tag)
    {
        case SP_TYPE_float:
            return simpla::data_model::DataType::create<float>();

        case SP_TYPE_double:
            return simpla::data_model::DataType::create<double>();

        case SP_TYPE_int:
            return simpla::data_model::DataType::create<int>();

        case SP_TYPE_uint:
            return simpla::data_model::DataType::create<unsigned int>();

        case SP_TYPE_unsigned_int:
            return simpla::data_model::DataType::create<unsigned int>();

        case SP_TYPE_unsigned_long:
            return simpla::data_model::DataType::create<unsigned long>();

        case SP_TYPE_long:
            return simpla::data_model::DataType::create<long>();


        default:
            break;
    }

    return simpla::data_model::DataType();
}

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
                          int data_type_tag,
                          void *d,
                          int ndims,
                          size_type const *dims,
                          size_type const *start,
                          size_type const *stride,
                          size_type const *count,
                          size_type const *block,
                          size_type const *g_dims,
                          size_type const *g_start,
                          int flag)
{

    simpla::data_model::DataSet dset;

    dset.data_type = spDataTypeFromTag(data_type_tag, 0);
    dset.data_space = simpla::data_model::DataSpace(ndims, (g_dims != NULL) ? g_dims : dims);
    dset.data_space.select_hyperslab((g_start != NULL) ? g_start : start, stride, count, block);
    dset.memory_space = simpla::data_model::DataSpace(ndims, dims);
    dset.memory_space.select_hyperslab(start, stride, count, block);

    dset.data = std::shared_ptr<void>(d, simpla::tags::do_nothing());

    VERBOSE << os->self->write(url, dset, flag) << std::endl;
    return SP_SUCCESS;

}

int spMPIInitialize(int argc, char **argv)
{
    GLOBAL_COMM.init(argc, argv);
    return SP_SUCCESS;
};

int spMPIFinalize()
{
    GLOBAL_COMM.close();
    return SP_SUCCESS;
}

MPI_Comm spMPIComm() { return GLOBAL_COMM.comm(); }

size_type spMPIGenerateObjectId() { return (GLOBAL_COMM.generate_object_id()); }

int spMPIBarrier()
{
    GLOBAL_COMM.barrier();
    return SP_SUCCESS;
}

int spMPIRank() { return GLOBAL_COMM.rank(); }

int spMPISize() { return GLOBAL_COMM.num_of_process(); }

int spMPITopology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord)
{
    return GLOBAL_COMM.topology(mpi_topo_ndims, mpi_topo_dims, periods, mpi_topo_coord);
};
//
//int spRandomMultiNormalDistributionInCell(size_type const *min,
//                                          size_type const *max,
//                                          size_type const *strides,
//                                          unsigned int pic,
//                                          Real *rx,
//                                          Real *ry,
//                                          Real *rz,
//                                          Real *vx,
//                                          Real *vy,
//                                          Real *vz)
//{
//    std::mt19937 rnd_gen(6);
//
//    size_t number = (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]) * pic;
//
//
//    rnd_gen.discard(number);
//
//    rectangle_distribution<3> x_dist;
//
//    multi_normal_distribution<3> v_dist;
//
//
//    for (size_type i = min[0]; i < max[0]; ++i)
//        for (size_type j = min[1]; j < max[1]; ++j)
//            for (size_type k = min[2]; k < max[2]; ++k)
//            {
//                size_type s = (i * strides[0] + j * strides[1] + k * strides[2]);
//                for (int l = 0; l < pic; ++l)
//                {
//                    Real x[3], v[3];
//                    x_dist(rnd_gen, &x[0]);
//                    v_dist(rnd_gen, &v[0]);
//
//                    rx[s + l] = x[0];
//                    ry[s + l] = x[1];
//                    rz[s + l] = x[2];
//
//                    vx[s + l] = v[0];
//                    vy[s + l] = v[1];
//                    vz[s + l] = v[2];
//                }
//
//            }
//
//}
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


