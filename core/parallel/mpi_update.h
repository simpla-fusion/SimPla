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
	int remote;
	int send_tag;
	int recv_tag;
	MPIDataType send_type;
	MPIDataType recv_type;
};
void make_send_recv_list(DataSpace const & dataspace, DataType const & datatype,
		size_t const * pghost_width, std::vector<send_recv_s> *res);

void sync_update_dataset(DataSet * dset, size_t const * ghost_width = nullptr,
		std::vector<MPI_Request> * requests = nullptr);

void sync_update_continue(std::vector<send_recv_s> const &, void * data,
		std::vector<MPI_Request> * requests = nullptr);

typedef std::tuple<int, // remote id
		int, //tag
		size_t, // size
		std::shared_ptr<void> //data
> send_recv_buffer_s;

void sync_update_unordered(std::vector<send_recv_buffer_s> const & send_buffer,
		std::vector<send_recv_buffer_s> * recv_buffer,
		std::vector<MPI_Request> * requests = nullptr);

}  // namespace simpla

#endif /* CORE_PARALLEL_MPI_UPDATE_H_ */
