/*
 * @file mpi_update.h
 *
 *  Created on: 2014年11月10日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_MPI_UPDATE_H_
#define CORE_PARALLEL_MPI_UPDATE_H_
#include "mpi_comm.h"
#include "mpi_aux_functions.h"
#include "mpi_datatype.h"

namespace simpla
{
class DataSet;
class DataSpace;

struct mpi_send_recv_s
{
	int dest;
	int send_tag;
	int recv_tag;
	MPIDataType send_type;
	MPIDataType recv_type;
};
void make_send_recv_list(int object_id, DataType const & datatype, int ndims,
		size_t const * l_dims,
		std::vector<mpi_ghosts_shape_s> const & ghost_shape,
		std::vector<mpi_send_recv_s> *res);

void sync_update_continue(std::vector<mpi_send_recv_s> const &, void * data,
		std::vector<MPI_Request> * requests = nullptr);


void wait_all_request(std::vector<MPI_Request> *requests);

std::tuple<int, int, int> get_mpi_tag(int obj_id, int const * coord);

struct mpi_send_recv_buffer_s
{
	int dest;

	int send_tag;

	int recv_tag;

	size_t send_size;

	size_t recv_size;

	MPIDataType datatype;

	std::shared_ptr<void> send_data;
	std::shared_ptr<void> recv_data;
};

void sync_update_varlength(std::vector<mpi_send_recv_buffer_s>* send_buffer,
		std::vector<MPI_Request> * requests = nullptr);

}  // namespace simpla

#endif /* CORE_PARALLEL_MPI_UPDATE_H_ */
