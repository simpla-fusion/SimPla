//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPMPI_H
#define SIMPLA_SPMPI_H

#include <mpi.h>
#include "spDataType.h"

MPI_Datatype spDataTypeMPIType(struct spDataType_s const *);

int spMPIInitialize(int argc, char **argv);

int spMPIFinalize();

MPI_Comm spMPIComm();

size_type spMPIGenerateObjectId();

int spMPIBarrier();

int spMPIRank();

int spMPISize();

int spMPITopology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord);

size_type spMPIPrefixSums(size_type v);

size_type spMPISum(size_type v);

struct spMPICartUpdater_s;

typedef struct spMPICartUpdater_s spMPICartUpdater;

int spMPICartUpdateNdArrayHalo(int num_of_buffer, void **buffers, const spDataType *ele_type, int ndims,
                               const size_type *dims, const size_type *start, const size_type *,
                               const size_type *count, const size_type *, int mpi_sync_start_dims);

int spMPICartUpdaterCreateDA(spMPICartUpdater **updater, const spDataType *data_desc, int ndims,
                             const size_type *shape, const size_type *start, const size_type *stride,
                             const size_type *count, const size_type *block, int mpi_sync_start_dims);

int spMPICartUpdaterDestroy(spMPICartUpdater **updater);

int spMPICartUpdate(spMPICartUpdater const *updater, void *buffer);

int spMPICartUpdateAll(spMPICartUpdater const *updater, int num_of_buffer, void **buffers);

int spMPICartUpdateNdArrayHalo2(int num_of_buffer,
                                void **buffers,
                                const spDataType *data_desc,
                                int ndims,
                                const size_type *shape,
                                const size_type *start,
                                const size_type *stride,
                                const size_type *count,
                                const size_type *block,
                                int mpi_sync_start_dims);
#endif //SIMPLA_SPMPI_H
