//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPMPI_H
#define SIMPLA_SPMPI_H

#include <mpi.h>
#include "sp_config.h"

int spMPIInitialize(int argc, char **argv);

int spMPIFinalize();

MPI_Comm spMPIComm();

size_type spMPIGenerateObjectId();

int spMPIBarrier();

int spMPIRank();

int spMPISize();

int spMPITopology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord);

int spMPIPrefixSum(size_type *offset, size_type *p_count);

struct spMPICartUpdater_s;

typedef struct spMPICartUpdater_s spMPICartUpdater;

int spMPICartUpdaterCreate(spMPICartUpdater **updater,
                           MPI_Comm comm,
                           int data_type_tag,
                           int mpi_sync_start_dims,
                           int ndims,
                           const size_type *shape,
                           const size_type *start,
                           const size_type *stride,
                           const size_type *count,
                           const size_type *block,
                           const size_type *bucket_start,
                           const size_type *bucket_count,
                           const size_type *sorted_idx);

int spMPICartUpdaterDestroy(spMPICartUpdater **updater);

int spMPICartUpdate(spMPICartUpdater const *updater, void *buffer);

int spMPICartUpdateAll(spMPICartUpdater const *updater, int num_of_buffer, void **buffers);


#endif //SIMPLA_SPMPI_H
