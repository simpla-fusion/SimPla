/*
 * spIO.cpp
 *
 *  Created on: 2016年6月18日
 *      Author: salmon
 */

#include <memory>
#include <cassert>
#include "../src/io/IO.h"
//#include "../src/parallel/DistributedObject.h"
#include "../src/parallel/MPIComm.h"
#include "../src/data_model/DataSet.h"
#include "../src/data_model/DataType.h"
#include "../src/data_model/DataSpace.h"

extern "C"
{
#include "sp_capi.h"
};

#include "parallel/MPIDataType.h"

using namespace simpla;


struct spDataType_s
{
    simpla::data_model::DataType self;

    simpla::MPIDataType m_mpi_type_;

};

int spDataTypeCreate(spDataType **dtype, int type_tag, size_type s)
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

size_type spDataTypeSizeInByte(spDataType const *dtype) { return dtype->self.size_in_byte(); }

int spDataTypeIsValid(spDataType const *dtype) { return dtype->self.is_valid(); }

int spDataTypeExtent(spDataType *dtype, int rank, const size_type *d)
{
    dtype->self.extent(rank, d);
    return SP_SUCCESS;
}
int spDataTypeAdd(spDataType *dtype, size_type offset, char const *name, spDataType const *other)
{
    dtype->self.push_back(other->self, name, offset);

    return SP_SUCCESS;
}

int spDataTypeAddArray(spDataType *dtype,
                       size_type offset,
                       char const *name,
                       int type_tag,
                       size_type n,
                       size_type const *dims)
{
    spDataType *ele;

    spDataTypeCreate(&ele, type_tag, 0);

    if (dims == nullptr) { spDataTypeExtent(ele, 1, &n); } else { spDataTypeExtent(ele, n, dims); }

    spDataTypeAdd(dtype, offset, name, ele);

    spDataTypeDestroy(&ele);

    return SP_SUCCESS;
}

int spDataTypeUpdate(spDataType *dtype)
{
    if (dtype->m_mpi_type_.type() == MPI_DATATYPE_NULL)
    {
        dtype->m_mpi_type_ = simpla::MPIDataType::create((dtype)->self);

    }
    return SP_SUCCESS;
}

MPI_Datatype const *spDataTypeMPIType(struct spDataType_s const *dtype) { return &(dtype->m_mpi_type_.type()); };

struct spDataSpace_s { simpla::data_model::DataSpace self; };

typedef struct spDataSpace_s spDataSpace;

void spDataSpaceCreateSimple(spDataSpace **, int ndims, int const *dims) {}

void spDataSpaceCreateUnordered(spDataSpace **, int num) {}

void spDataSpaceDestroy(spDataSpace **) {}

void spDataSpaceSelectHyperslab(spDataSpace *, ptrdiff_t const *offset, int const *count) {}

struct spDataSet_s;

typedef struct spDataSet_s spDataSet;

void spDataSetCreate(spDataSet **, void *d, spDataType const *dtype, spDataSpace const *mspace,
                     spDataSpace const *fspace) {}

void spDataSetDestroy(spDataSet *) {}

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

int spIOStreamWrite(spIOStream *, const char *name, spDataSet const *) { return SP_SUCCESS; }

int spIOStreamRead(spIOStream *, const char *name, spDataSet const *) { return SP_SUCCESS; }

int spIOStreamWriteSimple(spIOStream *os,
                          const char *url,
                          struct spDataType_s const *d_type,
                          void *d,
                          int ndims,
                          size_type const *dims,
                          size_type const *start,
                          size_type const *stride,
                          size_type const *count,
                          size_type const *block,
                          int flag)
{

    simpla::data_model::DataSet dset;

    dset.data_type = d_type->self;
    dset.data_space = simpla::data_model::DataSpace(ndims, (count != NULL) ? count : dims);
    dset.memory_space = simpla::data_model::DataSpace(ndims, dims);
    dset.memory_space.select_hyperslab(start, stride, count, block);

    dset.data = std::shared_ptr<void>(d, simpla::tags::do_nothing());

    MESSAGE << os->self->write(url, dset, flag) << std::endl;
    return SP_SUCCESS;

}
//
//struct spDistributedObject_s { simpla::parallel::DistributedObject self; };
//
//void spDistributeObjectCreate(struct spDistributedObject_s **obj) { *obj = new struct spDistributedObject_s; }
//
//void spDistributeObjectDestroy(struct spDistributedObject_s **obj)
//{
//    delete *obj;
//    *obj = 0x0;
//}
//
//void spDistributeObjectNonblockingSync(struct spDistributedObject_s *obj) { obj->self.sync(); }
//
//void spDistributeObjectWait(struct spDistributedObject_s *obj) { obj->self.wait(); }
//
//void hdf5_write_field(const char *url, void *d, int ndims, size_type const *dims, const size_type *offset,
//                      size_type const *count, int flag)
//{
//    simpla::data_model::DataSet dset;
//    simpla::data_model::DataSpace d_space(ndims, &count[0]);
//    simpla::data_model::DataSpace m_space(ndims, &dims[0]);
//    m_space.select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
//    dset.memory_space = m_space;
//    dset.data_space = d_space;
//    dset.data_type = simpla::data_model::DataType::create<Real>();
//    dset.data = std::shared_ptr<void>(reinterpret_cast<void *>(d), simpla::tags::do_nothing());
//
////	simpla::io::write(url, dset, id);
//
//}
//
//void spDistributedObjectAddSendLink(spDistributedObject *, int id, const ptrdiff_t offset[3], const spDataSet *) {}
//
//void spDistributedObjectAddRecvLink(spDistributedObject *, int id, const ptrdiff_t offset[3], spDataSet *) {}
//
//int spDistributedObjectIsReady(spDistributedObject const *) { return true; }

void spMPIInitialize(int argc, char **argv) { GLOBAL_COMM.init(argc, argv); };

void spMPIFinialize() { GLOBAL_COMM.close(); }

MPI_Comm spMPIComm() { return GLOBAL_COMM.comm(); }

MPI_Info spMPIInfo() { return GLOBAL_COMM.info(); }

void spMPIBarrier() { return GLOBAL_COMM.barrier(); }

int spMPIIsValid() { return (int) (GLOBAL_COMM.is_valid()); }

int spMPIProcessNum() { return (GLOBAL_COMM.process_num()); }

int spMPIRank() { return GLOBAL_COMM.process_num(); }

int spMPINumOfProcess() { return (GLOBAL_COMM.num_of_process()); }

int spMPISize() { return GLOBAL_COMM.num_of_process(); }

size_type spMPIGenerateObjectId() { return (GLOBAL_COMM.generate_object_id()); }

int spMPITopologyNumOfDims() { return GLOBAL_COMM.topology_num_of_dims(); }

int const *spMPITopologyDims() { return GLOBAL_COMM.topology_dims(); }

int spMPITopologyNumOfNeighbours() { return GLOBAL_COMM.topology_num_of_neighbours(); }

int const *spMPITopologyNeighbours() { return GLOBAL_COMM.topology_neighbours(); }

void spMPITopologyCoordinate(int rank, int *d) { GLOBAL_COMM.topology_coordinate(rank, d); }

int spMPITopologyRank(int const *d) { return GLOBAL_COMM.get_rank(d); };


