/**
 * @file spSimPlaWrap.h
 *
 *  Created on: 2016年6月16日
 *      Author: salmon
 */

#ifndef SPSIMPLAWRAP_H_
#define SPSIMPLAWRAP_H_

#include "stddef.h"
#include "sp_config.h"

#include <mpi.h>
#include <H5Ipublic.h>
enum
{
    SP_TYPE_NULL, SP_TYPE_float, SP_TYPE_double, SP_TYPE_int, SP_TYPE_long, SP_TYPE_int64_t, SP_TYPE_OPAQUE
};

#ifndef USE_FLOAT_REAL
#   define SP_TYPE_Real SP_TYPE_double
#else
#   define SP_TYPE_Real SP_TYPE_float
#endif

#define SP_TYPE_MeshEntityId SP_TYPE_int64_t

enum
{
    SP_FILE_NEW = 1UL << 1, SP_FILE_APPEND = 1UL << 2, SP_FILE_BUFFER = (1UL << 3), SP_FILE_RECORD = (1UL << 4)
};

struct spDataType_s;

typedef struct spDataType_s spDataType;

int spDataTypeCreate(spDataType **, int type_tag);

int spDataTypeDestroy(spDataType **);

int spDataTypeCopy(spDataType *, spDataType const *);

size_type spDataTypeSizeInByte(spDataType const *dtype);

void spDataTypeSetSizeInByte(spDataType *dtype, size_type s);

int spDataTypeIsValid(spDataType const *);

int spDataTypeExtent(spDataType *, int rank, const size_type *d);

int spDataTypeAdd(spDataType *dtype, size_type offset, char const *name, spDataType const *other);

int spDataTypeAddArray(spDataType *dtype,
                       size_type offset,
                       char const *name,
                       int type_tag,
                       size_type n,
                       size_type const *dims);

int spDataTypeUpdate(spDataType *dtype);

MPI_Datatype const *spDataTypeMPIType(struct spDataType_s const *);

struct spDataSpace_s;

typedef struct spDataSpace_s spDataSpace;

void spDataSpaceCreateSimple(spDataSpace **, int ndims, int const *dims);

void spDataSpaceCreateUnordered(spDataSpace **, int num);

void spDataSpaceDestroy(spDataSpace **);

void spDataSpaceSelectHyperslab(spDataSpace *, ptrdiff_t const *offset, int const *count);

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

void spDistributedObjectAddSendLink(spDistributedObject *, int id, const ptrdiff_t offset[3], const spDataSet *);

void spDistributedObjectAddRecvLink(spDistributedObject *, int id, const ptrdiff_t offset[3], spDataSet *);

int spDistributedObjectIsReady(spDistributedObject const *);

struct spIOStream_s;

typedef struct spIOStream_s spIOStream;

int spIOStreamCreate(spIOStream **);

int spIOStreamDestroy(spIOStream **);

int spIOStreamPWD(spIOStream *, char *url);

int spIOStreamOpen(spIOStream *, const char *url);

int spIOStreamClose(spIOStream *);

int spIOStreamWrite(spIOStream *, const char *name, spDataSet const *);

int spIOStreamRead(spIOStream *, const char *name, spDataSet const *);

int spIOStreamWriteSimple(spIOStream *,
                          const char *name,
                          struct spDataType_s const *d_type,
                          void *d,
                          int ndims,
                          size_type const *dims,
                          size_type const *start,
                          size_type const *stride,
                          size_type const *count,
                          size_type const *block,
                          int flag);

void spMPIInitialize(int argc, char **argv);

void spMPIFinialize();

MPI_Comm spMPIComm();

MPI_Info spMPIInfo();

void spMPIBarrier();

int spMPIIsValid();

int spMPIProcessNum();
int spMPINumOfProcess();

int spMPIRank();
int spMPISize();

size_type spMPIGenerateObjectId();

int spMPITopologyNumOfDims();

int const *spMPITopologyDims();

int spMPITopologyNumOfNeighbours();

int const *spMPITopologyNeighbours();

void spMPITopologyCoordinate(int rank, int *);

int spMPITopologyRank(int const *);


#endif /* SPSIMPLAWRAP_H_ */
