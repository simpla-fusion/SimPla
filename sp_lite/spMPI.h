//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPMPI_H
#define SIMPLA_SPMPI_H

#include <mpi.h>
#include "sp_lite_config.h"

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

int spMPIUpdaterCreate(spMPIUpdater **updater);

int spMPIUpdaterDestroy(spMPIUpdater **updater);


int spMPIUpdaterDeploy(spMPIUpdater *updater, int mpi_sync_start_dims, int ndims,
                       const size_type *shape, const size_type *start, const size_type *stride,
                       const size_type *count, const size_type *block);


int spMPIUpdateHalo(spMPIUpdater *updater, int type_tag, void *);

int spMPIUpdateBucket(spMPIUpdater *updater, int type_tag, int num, void **data, size_type *bucket_start,
                      size_type *bucket_count, size_type *sorted_index, size_type *tail);

int spMPIUpdateIndexed(spMPIUpdater *updater, int type_tag, int num, void **data,
                       size_type const *send_size, size_type **send_index,
                       size_type const *recv_size, size_type **recv_index);


#endif //SIMPLA_SPMPI_H
