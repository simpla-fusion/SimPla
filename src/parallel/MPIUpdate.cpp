/**
 * @file mpi_update.cpp
 *
 * @date    2014-7-29  AM8:32:26
 * @author salmon
 */

#include "MPIUpdate.h"
#include "MPIDataType.h"
#include "DistributedObject.h"
#include "../data_model/DataSet.h"
#include "../gtl/Log.h"

namespace simpla { namespace parallel
{


/**
 * @param pos in {0,count} out {begin,shape}
 */
std::tuple<int, int> sync_global_location(MPIComm &mpi_comm, int count)
{
    int begin = 0;

    if (mpi_comm.is_valid())
    {
        int num_of_process = mpi_comm.num_of_process();

        int process_num = mpi_comm.process_num();

        MPIDataType m_type = MPIDataType::create<int>();

        std::vector<int> buffer;

        if (process_num == 0)
        {
            buffer.resize(num_of_process);
        }
        MPI_Barrier(mpi_comm.comm());

        MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, mpi_comm.comm());

        MPI_Barrier(mpi_comm.comm());

        if (process_num == 0)
        {
            for (int i = 1; i < num_of_process; ++i)
            {
                buffer[i] += buffer[i - 1];
            }
            buffer[0] = count;
            count = buffer[num_of_process - 1];

            for (int i = num_of_process - 1; i > 0; --i)
            {
                buffer[i] = buffer[i - 1];
            }
            buffer[0] = 0;
        }
        MPI_Barrier(mpi_comm.comm());
        MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, mpi_comm.comm());
        MPI_Barrier(mpi_comm.comm());
        MPI_Bcast(&count, 1, m_type.type(), 0, mpi_comm.comm());
        MPI_Barrier(mpi_comm.comm());

    }

    return std::make_tuple(begin, count);
}




//
//void wait_all_request(std::vector<MPI_Request> &requests)
//{
//	if (!requests.empty())
//	{
//		MPI_ERROR(MPI_Waitall(requests.size(), const_cast<MPI_Request *>(&(requests[0])), MPI_STATUSES_IGNORE));
//
//		requests.clear();
//	}
//}
//
//void sync_update_block(MPIComm &mpi_comm, std::vector<mpi_send_recv_block_s> const &send_recv_list,
//		void *m_data, std::vector<MPI_Request> &requests)
//{
//
//
//	for (auto const &item : send_recv_list)
//	{
//		MPI_Request req;
//
//		MPI_ERROR(MPI_Isend(m_data, 1, item.send_type.type(), item.dest,
//				item.send_tag, mpi_comm.comm(), &req));
//
//		requests.push_back(std::move(req));
//	}
//	for (auto const &item : send_recv_list)
//	{
//		MPI_Request req;
//
//		MPI_ERROR(MPI_Irecv(m_data, 1, item.recv_type.type(), item.dest,
//				item.recv_tag, mpi_comm.comm(), &req));
//
//		requests.push_back(std::move(req));
//	}
//
//
//}
//
//
//void sync_update_varlength(MPIComm &mpi_comm, DataType const &DataType,
//		std::vector<send_recv_buffer_s> &send_recv_buffer,
//		std::vector<MPI_Request> &requests)
//{
//
//	auto mpi_data_type = MPIDataType::create(DataType);
//
//	int dest, send_tag, recv_tag;
//
//	for (auto &item : send_recv_buffer)
//	{
//		std::tie(dest, send_tag, std::ignore) = mpi_comm.make_send_recv_tag(object_id,
//				&item.coord_shift[0]);
//
//		MPI_Request send_req;
//
//		MPI_ERROR(MPI_Isend(item.send_data.get(), item.send_size, mpi_data_type.type(), dest, send_tag, mpi_comm.comm(),
//				&send_req));
//
//		requests.push_back(std::move(send_req));
//
//	}
//
//	for (auto &item: send_recv_buffer)
//	{
//		std::tie(dest, std::ignore, recv_tag) = mpi_comm.make_send_recv_tag(object_id,
//				&item.coord_shift[0]);
//
//		MPI_Status status;
//
//		MPI_ERROR(MPI_Probe(dest, recv_tag, mpi_comm.comm(), &status));
//
//		// When probe returns, the status object has the size and other
//		// attributes of the incoming message. Get the size of the message
//		int recv_num = 0;
//
//		MPI_ERROR(MPI_Get_count(&status, mpi_data_type.type(), &recv_num));
//
//		if (recv_num == MPI_UNDEFINED)
//		{
//			RUNTIME_ERROR("Update Ghosts particle fail");
//		}
//
//		item.recv_data = sp_alloc_memory(recv_num * DataType.size());
//
//		item.recv_size = recv_num;
//
//		MPI_Request recv_req;
//		MPI_ERROR(
//				MPI_Irecv(item.recv_data.get(), item.recv_size, mpi_data_type.type(), > dest, recv_tag,
//						mpi_comm.comm(), &recv_req));
//
//		requests->push_back(std::move(recv_req));
//	}
//
//
//}
//
//
//void get_ghost_shape(int ndims, size_t const *l_offset,
//		size_t const *l_stride, size_t const *l_count, size_t const *l_block,
//		size_t const *ghost_width,
//		std::vector<dist_sync_connection> *dist_connect)
//{
//	dist_connect->clear();
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;
//
//	for (unsigned int tag = 0, tag_e = (1U << (ndims * 2)); tag < tag_e; ++tag)
//	{
//		nTuple<int, 3> coords_shift;
//
//		bool tag_is_valid = true;
//
//		for (int n = 0; n < ndims; ++n)
//		{
//			if (((tag >> (n * 2)) & 3UL) == 3UL)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//			coords_shift[n] = ((tag >> (n * 2)) & 3U) - 1;
//
//			switch (coords_shift[n])
//			{
//			case 0:
//				send_count[n] = l_count[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = l_count[n];
//				recv_offset[n] = l_offset[n];
//				break;
//			case -1: //left
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n];
//
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] - ghost_width[n];
//
//				break;
//			case 1: //right
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
//
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] + l_count[n];
//				break;
//			default:
//				tag_is_valid = false;
//				break;
//			}
//
//			if (send_count[n] == 0 || recv_count[n] == 0)
//			{
//				tag_is_valid = false;
//				break;
//			}
//
//		}
//
//		if (tag_is_valid
//				&& (coords_shift[0] != 0 || coords_shift[1] != 0
//				|| coords_shift[2] != 0))
//		{
//
//			dist_connect->emplace_back(dist_sync_connection {coords_shift,
//			                                                 send_offset, send_count, recv_offset, recv_count});
//		}
//	}
//
//}

}} //namespace simpla{namespace  parallel{

