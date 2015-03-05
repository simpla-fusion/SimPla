/*
 * @file mpi_update.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_MPI_UPDATE_H_
#define CORE_PARALLEL_MPI_UPDATE_H_
#include "mpi_comm.h"
#include "mpi_datatype.h"

namespace simpla
{
class DataSet;
class DataSpace;

struct send_recv_s
{
	int dest;
	int send_tag;
	int recv_tag;
	MPIDataType send_type;
	MPIDataType recv_type;
};
void make_send_recv_list(DataType const & datatype, int ndims,
		size_t const * l_dims,
		std::vector<DataSpace::ghosts_shape_s> const & ghost_shape,
		std::vector<send_recv_s> *res);

//void sync_update_dataset(DataSet * dset, size_t const * ghost_width = nullptr,
//		std::vector<MPI_Request> * requests = nullptr);

void sync_update_continue(std::vector<send_recv_s> const &, void * data,
		std::vector<MPI_Request> * requests = nullptr);

typedef std::tuple<int, // remote id
		int, //tag
		size_t, // size
		std::shared_ptr<void> //data
> send_recv_buffer_s;

struct send_buffer_s
{
	int dest;
	int tag;
	size_t size;
	std::shared_ptr<void> data;
};

struct recv_buffer_s
{
	int dest;
	int tag;
	size_t size;
	std::shared_ptr<void> data;
};

void sync_update_varlength(std::vector<send_buffer_s> const & send_buffer,
		std::vector<recv_buffer_s> * recv_buffer,
		std::vector<MPI_Request> * requests = nullptr);

}  // namespace simpla

#endif /* CORE_PARALLEL_MPI_UPDATE_H_ */
