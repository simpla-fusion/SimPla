//
// Created by salmon on 16-9-12.
//

#ifndef SIMPLA_SPMPI_H
#define SIMPLA_SPMPI_H
#include <mpi.h>
#include "spDataType.h"


int spMPIInitialize(int argc, char **argv);

int spMPIFinalize();

MPI_Comm spMPIComm();

size_type spMPIGenerateObjectId();

int spMPIBarrier();

int spMPIRank();

int spMPISize();

int spMPITopology(int *mpi_topo_ndims, int *mpi_topo_dims, int *periods, int *mpi_topo_coord);

struct spMPIUpdater_s;

typedef struct spMPIUpdater_s spMPIUpdater;

MPI_Datatype spDataTypeMPIType(struct spDataType_s const *);

int spMPIUpdaterDestroy(spMPIUpdater **updater);

int spMPIUpdate(spMPIUpdater const *updater, void *buffer);

int spMPIUpdateAll(spMPIUpdater const *updater, int num_of_buffer, void **buffers);

int spMPIUpdateNdArrayHalo(int num_of_buffer, void **buffers, const spDataType *ele_type, int ndims,
                           const int *dims, const int *start, const int *,
                           const int *count, const int *, int mpi_sync_start_dims);

int spMPIUpdaterCreateDistArray(spMPIUpdater **updater,
                                MPI_Comm comm,
                                const spDataType *old_data,
                                int ndims,
                                const int *shape,
                                const int *start,
                                const int *stride,
                                const int *count,
                                const int *block,
                                int mpi_sync_start_dims);

int spMPIUpdaterCreateDistIndexed(spMPIUpdater **updater,
                                  MPI_Comm comm,
                                  const spDataType *old_data,
                                  int num_of_dims,
                                  int block_length,
                                  int const *send_count,
                                  MPI_Aint **send_index,
                                  int *send_disp,
                                  int const *recv_count,
                                  MPI_Aint **recv_index,
                                  int *recv_disp);

int spMPIUpdateNdArrayHalo2(int num_of_buffer,
                            void **buffers,
                            const spDataType *data_desc,
                            int ndims,
                            const int *shape,
                            const int *start,
                            const int *stride,
                            const int *count,
                            const int *block,
                            int mpi_sync_start_dims);

int spMPIPrefixSums(int v);

int spMPISum(int v);
#endif //SIMPLA_SPMPI_H
