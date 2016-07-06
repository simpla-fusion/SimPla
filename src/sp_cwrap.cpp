/*
 * spIO.cpp
 *
 *  Created on: 2016年6月18日
 *      Author: salmon
 */

#include <memory>
#include "../src/io/IO.h"
#include "../src/parallel/DistributedObject.h"

#include "../src/data_model/DataSet.h"
#include "../src/data_model/DataType.h"
#include "../src/data_model/DataSpace.h"

//#include "spSimPlaWrap.h"
//#include "sp_def.h"

using namespace simpla;



struct spDataType_s;
typedef struct spDataType_s spDataType;

void spDataTypeCreate(spDataType **) { }

void spDataTypeDestroy(spDataType **) { }

bool spDataTypeIsValid(spDataType const *) { }

void spDataTypeExtent(spDataType *, int rank, size_t const *d) { }

void spDataTypePushBack(spDataType *, spDataType const *, char const name[]) { }

struct spDataSpace_s;
typedef struct spDataSpace_s spDataSpace;

void spDataSpaceCreateSimple(spDataSpace **, int ndims, size_t const *dims) { }

void spDataSpaceCreateUnordered(spDataSpace **, size_t num) { }

void spDataSpaceDestroy(spDataSpace **) { }

void spDataSpaceSelectHyperslab(spDataSpace *, ptrdiff_t const *offset, size_t const *count) { }

struct spDataSet_s;
typedef struct spDataSet_s spDataSet;

void spDataSetCreate(spDataSet **, void *d, spDataType const *dtype, spDataSpace const *mspace,
                     spDataSpace const *fspace) { }

void spDataSetDestroy(spDataSet *) { }


struct spIOStream_s;
typedef struct spIOStream_s spIOStream;

void spIOStreamCreate(spIOStream **) { }

void spIOStreamDestroy(spIOStream **) { };

void spIOStreamOpen(spIOStream *, char const url[], int flag) { }

void spIOStreamClose(spIOStream *) { }

void spIOStreamWrite(spIOStream *, char const name[], spDataSet const *) { }

void spIOStreamRead(spIOStream *, char const name[], spDataSet const *) { }

void hdf5_write_field(spIOStream *, char const name[], //
                      void *d, int ndims, size_t const *dims, size_t const *start, size_t const *count, int flag) { }


struct spDistributedObject_s;
typedef struct spDistributedObject_s spDistributedObject;


struct spDistributedObject_s
{
    simpla::parallel::DistributedObject self;
};

void spDistributeObjectCreate(struct spDistributedObject_s **obj)
{
    *obj = new struct spDistributedObject_s;

}

void spDistributeObjectDestroy(struct spDistributedObject_s **obj)
{
    delete *obj;
    *obj = 0x0;
}

void spDistributeObjectNonblockingSync(struct spDistributedObject_s *obj)
{
    obj->self.sync();
}

void spDistributeObjectWait(struct spDistributedObject_s *obj)
{
    obj->self.wait();
}

void hdf5_write_field(char const url[], void *d, int ndims, size_type const *dims, ptrdiff_t const *offset,
                      size_type const *count, int flag)
{
    simpla::data_model::DataSet dset;
    data_model::DataSpace d_space(ndims, &count[0]);
    data_model::DataSpace m_space(ndims, &dims[0]);
    m_space.select_hyperslab(&offset[0], nullptr, &count[0], nullptr);
    dset.memory_space = m_space;
    dset.data_space = d_space;
    dset.data_type = data_model::DataType::create<Real>();
    dset.data = std::shared_ptr<void>(reinterpret_cast<void *>(d), tags::do_nothing());

//	simpla::io::write(url, dset, flag);

}


void spDistributedObjectAddSendLink(spDistributedObject *, size_t id, const ptrdiff_t offset[3], const spDataSet *) { }

void spDistributedObjectAddRecvLink(spDistributedObject *, size_t id, const ptrdiff_t offset[3], spDataSet *) { }

bool spDistributedObjectIsReady(spDistributedObject const *) { }