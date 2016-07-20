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
enum
{
    SP_TYPE_float, SP_TYPE_double, SP_TYPE_int, SP_TYPE_long, SP_TYPE_int64_t, SP_TYPE_OPAQUE
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

void spDataTypeCreate(spDataType **);

void spDataTypeDestroy(spDataType **);

int spDataTypeIsValid(spDataType const *);

void spDataTypeExtent(spDataType *, int rank, int const *d);

void spDataTypePushBack(spDataType *, spDataType const *, char const name[]);

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

void spIOStreamCreate(spIOStream **);

void spIOStreamDestroy(spIOStream **);

void spIOStreamPWD(spIOStream *, char url[]);

void spIOStreamOpen(spIOStream *, const char url[]);

void spIOStreamClose(spIOStream *);

void spIOStreamWrite(spIOStream *, char const name[], spDataSet const *);

void spIOStreamRead(spIOStream *, char const name[], spDataSet const *);

void spIOStreamWriteSimple(spIOStream *,
                           const char *name,
                           int d_type,
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

size_type spMPIGenerateObjectId();

void spMPIGetTopology(int *);

void spMPISetTopology(int *);

int spMPIGetNeighbour(int *);

void spMPICoordinate(int rank, int *);

int spMPIGetRank();

int spMPIGetRankCart(int const *);

void spMPIMakeSendRecvTag(size_type prefix, int const *offset, int *dest_id, int *send_tag, int *recv_tag);

#endif /* SPSIMPLAWRAP_H_ */
