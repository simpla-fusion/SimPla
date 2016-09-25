//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPMPI_H
#define SIMPLA_SPMPI_H

#include <mpi.h>
#include "sp_config.h"

struct spMPIUpdater_s;

typedef struct spMPIUpdater_s spMPIUpdater;

int spMPIInitialize(int argc, char **argv);

int spMPIFinalize();

MPI_Comm spMPIComm();

size_type spMPIGenerateObjectId();

int spMPIBarrier();

int spMPIRank();

int spMPISize();

int spMPITopology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord);

int spMPIPrefixSum(size_type *offset, size_type *p_count);

int spMPIUpdaterCreate(spMPIUpdater **updater, size_type size, int type_tag);

int spMPIUpdaterDestroy(spMPIUpdater **updater);

struct spMPIHaloUpdater_s;

typedef struct spMPIHaloUpdater_s spMPIHaloUpdater;

int spMPIHaloUpdaterCreate(spMPIHaloUpdater **updater, int type_tag);


int spMPIHaloUpdaterDeploy(spMPIHaloUpdater *updater, int mpi_sync_start_dims, int ndims,
                           const size_type *shape, const size_type *start, const size_type *stride,
                           const size_type *count, const size_type *block);


int spMPIHaloUpdaterDestroy(spMPIHaloUpdater **updater);

int spMPIHaloUpdate(spMPIHaloUpdater *updater, void *);

int spMPIHaloUpdateAll(spMPIHaloUpdater *updater, int num_of_buffer, void **buffers);


struct spMPIBucketUpdater_s;

typedef struct spMPIBucketUpdater_s spMPIBucketUpdater;

int spMPIBucketUpdaterCreate(spMPIBucketUpdater **updater, int type_tag);

int spMPIBucketUpdaterDestroy(spMPIBucketUpdater **updater);

int spMPIBucketUpdaterDeploy(spMPIBucketUpdater *updater, const size_type *shape, const size_type *start,
                             const size_type *count);

int spMPIBucketUpdaterSetup(spMPIBucketUpdater *updater,
                            const size_type *bucket_start,
                            const size_type *bucket_count,
                            const size_type *index);


int spMPIBucketUpdate(spMPIBucketUpdater *updater, void *buffer);

int spMPIBucketUpdateAll(spMPIBucketUpdater *updater, int num_of_buffer, void **buffers);


#endif //SIMPLA_SPMPI_H
