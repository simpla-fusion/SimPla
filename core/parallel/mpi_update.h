/**
 * @file mpi_update.h
 *
 *  Created on: 2014-11-10
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_MPI_UPDATE_H_
#define CORE_PARALLEL_MPI_UPDATE_H_

#include "mpi_comm.h"
#include "mpi_datatype.h"

namespace simpla { namespace parallel
{

/**
 * @param   in count out {begin,total}
 */

std::tuple<int, int> sync_global_location(MPIComm &mpi_comm, int count);


void wait_all_request(std::vector<MPI_Request> &requests);

std::tuple<int, int, int> get_mpi_tag(int obj_id, int const *coord);



//template<typename Integral>
//std::tuple<Integral, Integral> sync_global_location(MPIComm &mpi_comm, Integral count)
//{
//
//	auto res = sync_global_location(mpi_comm, static_cast<int>(count));
//
//	return std::make_tuple(static_cast<Integral>(std::get<0>(res)),
//			static_cast<Integral>(std::get<1>(res)));
//
//}
//struct mpi_send_recv_block_s
//{
//	int dest;
//	int send_tag;
//	int recv_tag;
//	MPIDataType send_type;
//	MPIDataType recv_type;
//};
//
//void sync_update_block(MPIComm const &mpi_comm, std::vector<mpi_send_recv_block_s> const &send_recv_list,
//		void *data, std::vector<MPI_Request> &requests);

//void sync_update_varlength(MPIComm &mpi_comm, DataType const &DataType,
//		std::vector<send_recv_buffer_s> &send_recv_buffer,
//		std::vector<MPI_Request> &requests);
//struct mpi_send_recv_buffer_s
//{
//	int dest;
//
//	int send_tag;
//
//	int recv_tag;
//
//	int send_size;
//
//	int recv_size;
//
//	MPIDataType DataType;
//
//	std::shared_ptr<void> send_data;
//	std::shared_ptr<void> recv_data;
//};
//
//
//struct dist_sync_connection
//{
//	nTuple<int, 3> coord_shift;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count;
//};
//
//void make_dist_connection(int ndims, size_t const *offset, size_t const *stride,
//		size_t const *count, size_t const *block, size_t const *ghost_width,
//		std::vector<dist_sync_connection> *dist_connect);
//
//void get_ghost_shape(int ndims, size_t const *l_offset,
//		size_t const *l_stride, size_t const *l_count, size_t const *l_block,
//		size_t const *ghost_width,
//		std::vector<dist_sync_connection> *dist_connect);

}}//namespace simpla{namespace parallel

#endif /* CORE_PARALLEL_MPI_UPDATE_H_ */
