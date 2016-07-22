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
#include <cassert>

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

//    {
//        int topo_type;
//        MPI_Topo_test(comm, &topo_type);
//        assert(topo_type == MPI_CART);
//    }
//
//    int mpi_topology_ndims = 0;
//
//    MPI_Datatype d_type[3];
//    MPI_Type_contiguous(3, MPI_INT, &d_type[0]);
//    MPI_Type_commit(&d_type[0]);
//
//    MPI_Type_vector(5, 1, 5, MPI_INT, &d_type[1]);
//    MPI_Type_commit(&d_type[1]);
//
//
//    MPI_Cartdim_get(comm, &mpi_topology_ndims);
//
//    int sizes[mpi_topology_ndims];
//    MPI_Aint send_displaces[mpi_topology_ndims * 2];
//    MPI_Aint recv_displaces[mpi_topology_ndims * 2];
//
//    for (int i = 0; i < mpi_topology_ndims; ++i)
//    {
//        int r0, r1;
//
//        MPI_Cart_shift(comm, i, 1, &r0, &r1);
//
//        MPI_Sendrecv(buffer + send_displaces[i * 2], 1, d_type[i], r0, tag,
//                     buffer + recv_displaces[i * 2], 1, d_type[i], r1, tag,
//                     comm, MPI_STATUS_IGNORE);
//
//        MPI_Sendrecv(buffer + send_displaces[i * 2 + 1], 1, d_type[i], r1, tag,
//                     buffer + recv_displaces[i * 2 + 1], 1, d_type[i], r0, tag,
//                     comm, MPI_STATUS_IGNORE);
//    }



//    MPI_Datatype x_dir_type;
//    MPI_Type_contiguous(3, MPI_INT, &x_dir_type);
//    MPI_Type_commit(&x_dir_type);
//    MPI_Datatype y_dir_type;
//    MPI_Type_vector(5, 1, 5, MPI_INT, &y_dir_type);
//    MPI_Type_commit(&y_dir_type);
//    int sizes[4] = {1, 1, 1, 1};
//    int s = sizeof(int);
//    MPI_Aint send_displs[4] = {5 * s, 15 * s, 6 * s, 8 * s};
//    MPI_Aint recv_displs[4] = {0 * s, 20 * s, 5 * s, 9 * s};
//    MPI_Datatype send_type[4] = {x_dir_type, x_dir_type, y_dir_type, y_dir_type};
//    MPI_Datatype recv_type[4] = {x_dir_type, x_dir_type, y_dir_type, y_dir_type};
//


//******************************************************************************************************************
// Async
//    int r0, r1;
//    MPI_Request requests[4];
//
//    MPI_Cart_shift(spMPIComm(), 0, 1, &r0, &r1);
//    MPI_Isend(buffer + 6,
//              1,
//              x_dir_type,
//              r0,
//              1,
//              spMPIComm(), &requests[0]);
//    MPI_Irecv(buffer + 1,
//              1,
//              x_dir_type,
//              r0,
//              1,
//              spMPIComm(), &requests[1]);
//
//    MPI_Isend(buffer + 16,
//              1,
//              x_dir_type,
//              r1,
//              1,
//              spMPIComm(), &requests[2]);
//    MPI_Irecv(buffer + 21,
//              1,
//              x_dir_type,
//              r1,
//              1,
//              spMPIComm(), &requests[3]);
//
//    MPI_Waitall(4, requests, MPI_STATUS_IGNORE);
//
//    MPI_Cart_shift(spMPIComm(), 1, 1, &r0, &r1);
//
//    MPI_Isend(buffer + 1,
//              1,
//              y_dir_type,
//              r0,
//              1,
//              spMPIComm(), &requests[0]);
//    MPI_Irecv(buffer + 0,
//              1,
//              y_dir_type,
//              r0,
//              1,
//              spMPIComm(), &requests[1]);
//
//    MPI_Isend(buffer + 3,
//              1,
//              y_dir_type,
//              r1,
//              1,
//              spMPIComm(), &requests[2]);
//    MPI_Irecv(buffer + 4,
//              1,
//              y_dir_type,
//              r1,
//              1,
//              spMPIComm(), &requests[3]);
//
//    MPI_Waitall(4, requests, MPI_STATUS_IGNORE);
//******************************************************************************************************************

//    int buffer[9] = {
//        0, 0, 0,
//        0, rank, 0,
//        0, 0, 0
//
//    };
//
//    int num_of_neighbour = spMPITopologyNumOfNeighbours();
//
//    MPI_Datatype x_dir_type;
//    MPI_Type_contiguous(1, MPI_INT, &x_dir_type);
//    MPI_Type_commit(&x_dir_type);
//    MPI_Datatype y_dir_type;
//    MPI_Type_vector(3, 1, 3, MPI_INT, &y_dir_type);
//    MPI_Type_commit(&y_dir_type);
//    int sizes[4] = {1, 1, 1, 1};
//    int s = sizeof(int);
//    MPI_Aint send_displs[4] = {3 * s, 3 * s, 1 * s, 1 * s};
//    MPI_Aint recv_displs[4] = {1 * s, 7 * s, 0 * s, 2 * s};
//    MPI_Datatype send_type[4] = {x_dir_type, x_dir_type, y_dir_type, y_dir_type};
//    MPI_Datatype recv_type[4] = {x_dir_type, x_dir_type, y_dir_type, y_dir_type};
//
////    MPI_Neighbor_alltoallw(buffer, sizes, send_displs, send_type,
////                           buffer, sizes, recv_displs, recv_type, spMPIComm());
//
//
//    int r0, r1;
//    MPI_Cart_shift(spMPIComm(), 0, 1, &r0, &r1);
//    MPI_Sendrecv(buffer + 4,
//                 1,
//                 x_dir_type,
//                 r0,
//                 1,
//                 buffer + 1,
//                 1,
//                 x_dir_type,
//                 r1,
//                 1,
//                 spMPIComm(),
//                 MPI_STATUS_IGNORE);
//    MPI_Cart_shift(spMPIComm(), 0, -1, &r0, &r1);
//    MPI_Sendrecv(buffer + 4,
//                 1,
//                 x_dir_type,
//                 r0,
//                 1,
//                 buffer + 7,
//                 1,
//                 x_dir_type,
//                 r1,
//                 1,
//                 spMPIComm(),
//                 MPI_STATUS_IGNORE);
//
//
//    MPI_Cart_shift(spMPIComm(), 1, 1, &r0, &r1);
//    MPI_Sendrecv(buffer + 1,
//                 1,
//                 y_dir_type,
//                 r0,
//                 1,
//                 buffer + 0,
//                 1,
//                 y_dir_type,
//                 r1,
//                 1,
//                 spMPIComm(),
//                 MPI_STATUS_IGNORE);
//    MPI_Cart_shift(spMPIComm(), 1, -1, &r0, &r1);
//    MPI_Sendrecv(buffer + 1,
//                 1,
//                 y_dir_type,
//                 r0,
//                 1,
//                 buffer + 2,
//                 1,
//                 y_dir_type,
//                 r1,
//                 1,
//                 spMPIComm(),
//                 MPI_STATUS_IGNORE);
//
//
//    printf("\n"
//               "[%d/%d/%d] \t  %d,%d,%d  \n"
//               "           \t  %d,%d,%d  \n"
//               "           \t  %d,%d,%d  \n", rank, size, num_of_neighbour,
//           buffer[0], buffer[1], buffer[2],
//           buffer[3], buffer[4], buffer[5],
//           buffer[6], buffer[7], buffer[8]
//    );






//    int rank = spMPIRank();
//    int size = spMPISize();
//    int buffer[25] = {
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank,
//        rank, rank, rank, rank, rank
//    };
//    MPI_Datatype d_type[3];
//    MPI_Type_contiguous(3, MPI_INT, &d_type[0]);
//    MPI_Type_commit(&d_type[0]);
//
//    MPI_Type_vector(5, 1, 5, MPI_INT, &d_type[1]);
//    MPI_Type_commit(&d_type[1]);
//
////    MPI_Type_vector(5, 1, 5, MPI_INT, &d_type[2]);
////    MPI_Type_commit(&d_type[2]);
//
//    int sizes[3] = {1, 1, 1};
//    MPI_Aint send_displs[6] = {6, 16, 1, 3, 0, 0};
//    MPI_Aint recv_displs[6] = {1, 21, 0, 4, 0, 0};
//    int tag = 1;
//
//    for (int i = 0; i < 2; ++i)
//    {
//        int r0, r1;
//
//
//        MPI_Cart_shift(spMPIComm(), i, 1, &r0, &r1);
//
//        MPI_Sendrecv(buffer + send_displs[i * 2], sizes[i], d_type[i], r0, tag,
//                     buffer + recv_displs[i * 2], sizes[i], d_type[i], r1, tag,
//                     spMPIComm(), MPI_STATUS_IGNORE);
//
//        MPI_Sendrecv(buffer + send_displs[i * 2 + 1], sizes[i], d_type[i], r1, tag,
//                     buffer + recv_displs[i * 2 + 1], sizes[i], d_type[i], r0, tag,
//                     spMPIComm(), MPI_STATUS_IGNORE);
//    }
//
////    MPI_Cart_shift(spMPIComm(), 1, 1, &r0, &r1);
////
////    MPI_Sendrecv(buffer + 1, 1, y_dir_type, r0, tag,
////                 buffer + 0, 1, y_dir_type, r1, tag,
////                 spMPIComm(), MPI_STATUS_IGNORE);
////
////    MPI_Sendrecv(buffer + 3, 1, y_dir_type, r1, tag,
////                 buffer + 4, 1, y_dir_type, r0, tag,
////                 spMPIComm(), MPI_STATUS_IGNORE);
//
//
//    printf("\n"
//               "[%d/%d/%d] \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n"
//               "           \t  %d,%d,%d,%d,%d \n", rank, size, num_of_neighbour,
//           buffer[0], buffer[1], buffer[2], buffer[3], buffer[4],
//           buffer[5], buffer[6], buffer[7], buffer[8], buffer[9],
//           buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
//           buffer[15], buffer[16], buffer[17], buffer[18], buffer[19],
//           buffer[20], buffer[21], buffer[22], buffer[23], buffer[24]
//    );





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

