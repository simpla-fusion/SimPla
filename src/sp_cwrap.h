/**
 * @file spSimPlaWrap.h
 *
 *  Created on: 2016年6月16日
 *      Author: salmon
 */

#ifndef SPSIMPLAWRAP_H_
#define SPSIMPLAWRAP_H_

#include "stddef.h"

enum
{
    SP_TYPE_float, SP_TYPE_double, SP_TYPE_int, SP_TYPE_long, SP_TYPE_OPAQUE
};
#define SP_TYPE_Real SP_TYPE_float
enum
{
    SP_FILE_NEW = 1UL << 1, SP_FILE_APPEND = 1UL << 2, SP_FILE_BUFFER = (1UL << 3), SP_FILE_RECORD = (1UL << 4)
};

struct spDataType_s;
typedef struct spDataType_s spDataType;

void spDataTypeCreate(spDataType **);

void spDataTypeDestroy(spDataType **);

bool spDataTypeIsValid(spDataType const *);

void spDataTypeExtent(spDataType *, int rank, size_t const *d);

void spDataTypePushBack(spDataType *, spDataType const *, char const name[]);

struct spDataSpace_s;
typedef struct spDataSpace_s spDataSpace;

void spDataSpaceCreateSimple(spDataSpace **, int ndims, size_t const *dims);

void spDataSpaceCreateUnordered(spDataSpace **, size_t num);

void spDataSpaceDestroy(spDataSpace **);

void spDataSpaceSelectHyperslab(spDataSpace *, ptrdiff_t const *offset, size_t const *count);

struct spDataSet_s;
typedef struct spDataSet_s spDataSet;

void spDataSetCreate(spDataSet **, void *d, spDataType const *dtype, spDataSpace const *mspace,
                     spDataSpace const *fspace);

void spDataSetDestroy(spDataSet *);

struct spDistributedObject_s;
typedef struct spDistributedObject_s spDistributedObject;

void spDistributedObjectCreate(spDistributedObject **);

void spDistributedObjectDestroy(spDistributedObject **);

void spDistributedObjectNonblockingSync(spDistributedObject *);

void spDistributedObjectWait(spDistributedObject *);

void spDistributedObjectAddSendLink(spDistributedObject *, size_t id, const ptrdiff_t offset[3], const spDataSet *);

void spDistributedObjectAddRecvLink(spDistributedObject *, size_t id, const ptrdiff_t offset[3], spDataSet *);

bool spDistributedObjectIsReady(spDistributedObject const *);

struct spIOStream_s;
typedef struct spIOStream_s spIOStream;

void spIOStreamCreate(spIOStream **);

void spIOStreamDestroy(spIOStream **);

void spIOStreamOpen(spIOStream *, char const url[], int flag);

void spIOStreamClose(spIOStream *);

void spIOStreamWrite(spIOStream *, char const name[], spDataSet const *);

void spIOStreamRead(spIOStream *, char const name[], spDataSet const *);

void hdf5_write_field(spIOStream *, char const name[], //
                      void *d, int ndims, size_t const *dims, size_t const *start, size_t const *count, int flag);

#endif /* SPSIMPLAWRAP_H_ */
