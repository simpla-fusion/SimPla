/**
 * @file mpi_update.cpp
 *
 * @date    2014年7月29日  上午8:32:26
 * @author salmon
 */

#include "mpi_update.h"
#include "mpi_datatype.h"
#include "../dataset/dataset.h"
#include "../utilities/log.h"

namespace simpla
{
/**
 * @param pos in {0,count} out {begin,shape}
 */
std::tuple<int, int> sync_global_location(int count)
{

	int begin = 0;

	if ( GLOBAL_COMM.is_valid() && GLOBAL_COMM.num_of_process() > 1)
	{
		auto comm = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.num_of_process();
		int porcess_number = GLOBAL_COMM.process_num( );

		MPIDataType m_type =MPIDataType::create<int>();

		std::vector<int> buffer;

		if (porcess_number == 0)
		buffer.resize(num_of_process);
		MPI_Barrier(comm);

		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, comm);

		MPI_Barrier(comm);

		if (porcess_number == 0)
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
		MPI_Barrier(comm);
		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, comm);
		MPI_Barrier(comm);
		MPI_Bcast(&count, 1, m_type.type(), 0, comm);
		MPI_Barrier(comm);

	}

	return std::make_tuple(begin, count);

}
void get_ghost_shape(int ndims, size_t const *l_offset,
		size_t const *l_stride, size_t const *l_count, size_t const *l_block,
		size_t const *ghost_width,
		std::vector<mpi_ghosts_shape_s> *send_recv_list)
{
	send_recv_list->clear();

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned int tag = 0, tag_e = (1UL << (ndims * 2)); tag < tag_e; ++tag)
	{
		nTuple<int, 3> coords_shift;

		bool tag_is_valid = true;

		for (int n = 0; n < ndims; ++n)
		{
			if (((tag >> (n * 2)) & 3UL) == 3UL)
			{
				tag_is_valid = false;
				break;
			}

			coords_shift[n] = ((tag >> (n * 2)) & 3UL) - 1;

			switch (coords_shift[n])
			{
			case 0:
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
				break;
			case -1: //left

				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] - ghost_width[n];

				break;
			case 1: //right
				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];

				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] + l_count[n];
				break;
			default:
				tag_is_valid = false;
				break;
			}

			if (send_count[n] == 0 || recv_count[n] == 0)
			{
				tag_is_valid = false;
				break;
			}

		}

		if (tag_is_valid
				&& (coords_shift[0] != 0 || coords_shift[1] != 0
						|| coords_shift[2] != 0))
		{

			send_recv_list->emplace_back(mpi_ghosts_shape_s { coords_shift,
					send_offset, send_count, recv_offset, recv_count });
		}
	}

}

void make_send_recv_list(int object_id, DataType const & datatype, int ndims,
		size_t const * l_dims,
		std::vector<mpi_ghosts_shape_s> const & ghost_shape,
		std::vector<mpi_send_recv_s> *res)
{

	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	for (auto const & item : ghost_shape)
	{
		int dest, send_tag, recv_tag;

		std::tie(dest, send_tag, recv_tag) = GLOBAL_COMM.make_send_recv_tag(object_id,
				&item.coord_shift[0]);

		res->emplace_back(

				mpi_send_recv_s {

				dest,

				send_tag,

				recv_tag,

				MPIDataType::create(datatype, ndims, l_dims,
						&item.send_offset[0], nullptr, &item.send_count[0],
						nullptr),

				MPIDataType::create(datatype, ndims, l_dims,
						&item.recv_offset[0], nullptr, &item.recv_count[0],
						nullptr) }

				);

	}

//	if (pghost_width == nullptr /*|| mpi_comm.num_of_process() <= 1*/)
//	{
//		return;
//	}
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> ghost_width;
//
//	ghost_width = pghost_width;
//
//	size_t ndims = 3;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_dims, g_offset, g_count;
//
//	std::tie(ndims, g_dims, g_offset, std::ignore, g_count, std::ignore) =
//			dataspace.global_shape();
//
//	nTuple<size_t, 3> l_dims, l_offset, l_count;
//
//	std::tie(std::ignore, l_dims, l_offset, std::ignore, l_count, std::ignore) =
//			dataspace.shape();
//
//	auto mpi_topology = mpi_comm.Topology();
//
//	for (int n = 0; n < ndims; ++n)
//	{
//		if (mpi_topology[n] > (l_dims[n]))
//		{
//			RUNTIME_ERROR(
//					"DataSpace decompose fail! Dimension  is smaller than process grid. "
//							"[dimensions= " + value_to_string(l_dims)
//							+ ", process dimensions="
//							+ value_to_string(mpi_topology) + ", ghost_width"
//							+ value_to_string(ghost_width));
//		}
//	}
//
//	int count = 0;
//
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
//	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;
//
//	for (unsigned int tag = 0, tag_e = (1UL << (ndims * 2)); tag < tag_e; ++tag)
//	{
//		nTuple<int, MAX_NDIMS_OF_ARRAY> coords_shift;
//
//		bool tag_is_valid = false;
//
//		for (int n = 0; n < ndims; ++n)
//		{
//			coords_shift[n] = ((tag >> (n * 2)) & 3UL) - 1;
//
//			switch (coords_shift[n])
//			{
//			case 0:
//				send_count[n] = l_count[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = l_count[n];
//				recv_offset[n] = l_offset[n];
//				break;
//			case -1:
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n];
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] - ghost_width[n];
//				tag_is_valid = true;
//				break;
//
//			case 1:
//
//				send_count[n] = ghost_width[n];
//				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
//				recv_count[n] = ghost_width[n];
//				recv_offset[n] = l_offset[n] + l_count[n];
//				tag_is_valid = true;
//				break;
//			}
//
//			if (send_count[n] == 0 || recv_count[n] == 0)
//			{
//				tag_is_valid = false;
//				break;
//			}
//		}
//
//		if (tag_is_valid)
//		{
//
//			res->emplace_back(
//
//					mpi_send_recv_s {
//
//					mpi_comm.get_neighbour(coords_shift),
//
//					static_cast<int>(tag),
//
//					static_cast<int>(tag),
//
//					MPIDataType::create(datatype, ndims, &l_dims[0],
//							&send_offset[0], nullptr, &send_count[0], nullptr),
//
//					MPIDataType::create(datatype, ndims, &l_dims[0],
//							&recv_offset[0], nullptr, &recv_count[0], nullptr) }
//
//					);
//		}
//
//	}
}

//void sync_update_dataset(DataSet * dset, size_t const * ghost_width,
//		std::vector<MPI_Request> *requests)
//{
//	std::vector<mpi_send_recv_s> s_r_list;
//
//	make_send_recv_list(dset->dataspace, dset->datatype, ghost_width,
//			&s_r_list);
//
//	sync_update_continue(s_r_list, dset->data.get(), requests);
//}

void wait_all_request(std::vector<MPI_Request> *requests)
{
	if (requests != nullptr && requests->size() > 0)
	{
		MPI_ERROR(MPI_Waitall( requests->size(), //
				const_cast<MPI_Request*>(&(*requests)[0]),//
				MPI_STATUSES_IGNORE));
		requests->clear();
	}
}

void sync_update_continue(std::vector<mpi_send_recv_s> const & send_recv_list,
		void * data, std::vector<MPI_Request> *requests)
{
	bool is_async = true;
	if (requests == nullptr)
	{
		is_async = false;
		requests = new std::vector<MPI_Request>;
	}

	MPI_Comm mpi_comm = SingletonHolder<simpla::MPIComm>::instance().comm();

	for (auto const & item : send_recv_list)
	{

		{
			MPI_Request req;

			MPI_ERROR(
					MPI_Isend(data, 1, item.send_type.type(), item.dest,
							item.send_tag, mpi_comm, &req));

			requests->push_back(std::move(req));
		}

		{
			MPI_Request req;
			MPI_ERROR(
					MPI_Irecv(data, 1, item.recv_type.type(), item.dest,
							item.recv_tag, mpi_comm, &req));

			requests->push_back(std::move(req));
		}

	}

	if (!is_async)
	{
		wait_all_request(requests);

		delete requests;
	}

}

void sync_update_varlength(
		std::vector<mpi_send_recv_buffer_s> * send_recv_buffer,
		std::vector<MPI_Request> *requests)
{
	bool is_async = true;
	if (requests == nullptr)
	{
		is_async = false;
		requests = new std::vector<MPI_Request>;
	}

	MPI_Comm mpi_comm = SingletonHolder<simpla::MPIComm>::instance().comm();

	for (auto it = send_recv_buffer->begin(), ie = send_recv_buffer->end();
			it != ie; ++it)
	{

		MPI_Request req;

		MPI_ERROR(
				MPI_Isend(it->send_data.get(), it->send_size,
						it->datatype.type(), it->dest, it->send_tag, mpi_comm,
						&req));
		requests->push_back(std::move(req));

	}

	for (auto it = send_recv_buffer->begin(), ie = send_recv_buffer->end();
			it != ie; ++it)
	{

		MPI_Status status;

		MPI_ERROR(MPI_Probe(it->dest, it->recv_tag, mpi_comm, &status));

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int recv_num = 0;

		MPI_ERROR(MPI_Get_count(&status, it->datatype.type(), &recv_num));

		if (recv_num == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}

		it->recv_data = sp_alloc_memory(recv_num * it->datatype.size());

		it->recv_size = recv_num;
		{
			MPI_Request req;
			MPI_ERROR(
					MPI_Irecv(it->recv_data.get(),
							it->recv_size, //
							it->datatype.type(), it->dest, it->recv_tag,
							mpi_comm, &req));

			requests->push_back(std::move(req));
		}
	}

	if (!is_async)
	{
		wait_all_request(requests);
		delete requests;
	}

}

}
// namespace simpla
