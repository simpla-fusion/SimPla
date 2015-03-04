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

void make_send_recv_list(DataSpace const & dataspace, DataType const & datatype,
		size_t const * pghost_width, std::vector<send_recv_s> *res)
{
	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	if (pghost_width == nullptr || mpi_comm.num_of_process() <= 1)
	{
		return;
	}

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> ghost_width;

	ghost_width = pghost_width;

	size_t ndims = 3;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> g_dims, g_offset, g_count;

	std::tie(ndims, g_dims, g_offset, std::ignore, g_count, std::ignore) =
			dataspace.global_shape();

	nTuple<size_t, 3> l_dims, l_offset, l_count;

	std::tie(std::ignore, l_dims, l_offset, std::ignore, l_count, std::ignore) =
			dataspace.shape();

	auto mpi_topology = mpi_comm.topology();

	for (int n = 0; n < ndims; ++n)
	{
		if (mpi_topology[n] > (l_dims[n]))
		{
			RUNTIME_ERROR(
					"DataSpace decompose fail! Dimension  is smaller than process grid. "
							"[dimensions= " + value_to_string(l_dims)
							+ ", process dimensions="
							+ value_to_string(mpi_topology) + ", ghost_width"
							+ value_to_string(ghost_width));
		}
	}

	int count = 0;

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned long s = 0, s_e = (1UL << (ndims * 2)); s < s_e; ++s)
	{
		nTuple<int, MAX_NDIMS_OF_ARRAY> coords_shift;

		bool is_duplicate = false;

		for (int n = 0; n < ndims; ++n)
		{

			coords_shift[n] = ((s >> (n * 2)) & 3UL) - 1;

			if (coords_shift[n] == 0)
			{
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
			}
			else
			{
				if (ghost_width[n] == 0)
				{
					is_duplicate = true;
					break;
				}
				else if (coords_shift[n] == -1)
				{
					send_count[n] = ghost_width[n];
					send_offset[n] = l_offset[n];
					recv_count[n] = ghost_width[n];
					recv_offset[n] = (l_offset[n] - ghost_width[n] + l_dims[n])
							% l_dims[n];
				}
				else if (coords_shift[n] == 1)
				{
					send_count[n] = ghost_width[n];
					send_offset[n] = (l_offset[n] + l_count[n] - ghost_width[n]
							+ l_dims[n]) % l_dims[n];
					recv_count[n] = ghost_width[n];
					recv_offset[n] = (l_offset[n] + l_count[n] + l_dims[n])
							% l_dims[n];
				}
				else
				{
					is_duplicate = true;
					break;
				}
			}

		}

		if (!is_duplicate)
		{

			res->emplace_back(

					send_recv_s
					{

					mpi_comm.get_neighbour(coords_shift),

					static_cast<int>(send_offset[0]),

					static_cast<int>(recv_offset[0]),

					MPIDataType::create(datatype, ndims, &l_dims[0],
							&send_offset[0], nullptr, &send_count[0], nullptr),

					MPIDataType::create(datatype, ndims, &l_dims[0],
							&recv_offset[0], nullptr, &recv_count[0], nullptr) }

					);
		}

	}
}

void sync_update_dataset(DataSet * dset, size_t const * ghost_width,
		std::vector<MPI_Request> *requests)
{
	std::vector<send_recv_s> s_r_list;

	make_send_recv_list(dset->dataspace, dset->datatype, ghost_width,
			&s_r_list);

	sync_update_continue(s_r_list, dset->data.get(), requests);
}

void sync_update_continue(std::vector<send_recv_s> const & send_recv_list,
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
		MPI_Request req1;

		MPI_ERROR(
				MPI_Isend(data, 1, item.send_type.type(), item.remote,
						item.send_tag, mpi_comm, &req1));

		requests->push_back(req1);

		MPI_Request req2;
		MPI_ERROR(
				MPI_Irecv(data, 1, item.recv_type.type(), item.remote,
						item.recv_tag, mpi_comm, &req2));

		requests->push_back(req2);
	}

	if (!is_async)
	{
		MPI_ERROR(
				MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE));
		delete requests;
	}

}

void sync_update_unordered(std::vector<send_recv_buffer_s> const & send_buffer,
		std::vector<send_recv_buffer_s> * recv_buffer,
		std::vector<MPI_Request> *requests)
{
	bool is_async = true;
	if (requests == nullptr)
	{
		is_async = false;
		requests = new std::vector<MPI_Request>;
	}

	MPIComm & global_comm = SingletonHolder<simpla::MPIComm>::instance();

	int num_of_reqs = send_buffer.size() + recv_buffer->size();

	requests->resize(num_of_reqs);

	MPI_Request * req_it = &(*requests)[0];

	std::map<int, std::pair<size_t, std::shared_ptr<void> > > out_buffer;

	int count = 0;

	for (auto it = send_buffer.begin(), ie = send_buffer.end(); it != ie; ++it)
	{
		int remote, send_tag;
		size_t mem_size;
		std::shared_ptr<void> data;

		std::tie(remote, send_tag, mem_size, data) = *it;

		MPI_ERROR(
				MPI_Isend(data.get(), mem_size, MPI_BYTE, remote, send_tag, global_comm.comm(), req_it));

		++req_it;

	}

	for (auto it = recv_buffer->begin(), ie = recv_buffer->end(); it != ie;
			++it)
	{
		int remote, recv_tag;

		std::tie(remote, recv_tag, std::ignore, std::ignore) = *it;

		MPI_Status status;

		MPI_ERROR(MPI_Probe(remote, recv_tag, global_comm.comm(), &status));

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int recv_mem_size = 0;

		MPI_ERROR(MPI_Get_count(&status, MPI_BYTE, &recv_mem_size));

		if (recv_mem_size == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}

		std::shared_ptr<void> data;
		data = sp_alloc_memory(recv_mem_size);

		MPI_ERROR(
				MPI_Irecv(data.get(), recv_mem_size, MPI_BYTE, remote, recv_tag, global_comm.comm(), req_it));

		std::get<2>(*it) = recv_mem_size;
		std::get<3>(*it) = data;
		++req_it;
	}

	if (!is_async)
	{
		MPI_Waitall(requests->size(), &(*requests)[0], MPI_STATUSES_IGNORE);
		delete requests;
	}

}

}
// namespace simpla
