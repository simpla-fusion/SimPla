/**
 * @file mpi_update.cpp
 *
 * @date    2014年7月29日  上午8:32:26
 * @author salmon
 */

#include "../utilities/log.h"
#include "../dataset/dataset.h"
#include "mpi_comm.h"
#include "mpi_datatype.h"
#include "mpi_update.h"
namespace simpla
{

std::vector<send_recv_s> decompose(DataSpace const & dataspace,
		DataType const & datatype)
{
	std::vector<send_recv_s> res;

	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	static constexpr size_t ndims = 3;

	auto ghost_width = dataspace.ghost_width();

	nTuple<size_t, 3> g_dims, g_offset, g_count;

	std::tie(std::ignore, g_dims, g_offset, g_count, std::ignore, std::ignore) =
			dataspace.global().shape();

	nTuple<size_t, 3> l_dims, l_offset, l_count;

	std::tie(std::ignore, l_dims, l_offset, l_count, std::ignore, std::ignore) =
			dataspace.local().shape();

	auto mpi_topology = mpi_comm.get_topology();

	for (int n = 0; n < ndims; ++n)
	{
		if (mpi_topology[n] < (l_dims[n] + ghost_width[n] * 2))
		{
			RUNTIME_ERROR(
					"DataSpace decompose fail! Dimension  is smaller than process grid. "
							"[dimensions= " + value_to_string(l_dims)
							+ ", process dimensions="
							+ value_to_string(mpi_topology));
		}
	}

	int count = 0;

	nTuple<size_t, ndims> send_count, send_offset;
	nTuple<size_t, ndims> recv_count, recv_offset;

	for (unsigned long s = 0, s_e = (1UL << (ndims * 2)); s < s_e; ++s)
	{
		nTuple<int, ndims> coords_shift;

		bool is_duplicate = false;

		for (int n = 0; n < ndims; ++n)
		{

			coords_shift[n] = ((s >> (n * 2)) & 3UL) - 1;

			switch (coords_shift[n])
			{
			case -1:
				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] - ghost_width[n];
				break;
			case 0:
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
				break;
			case 1:
				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] + l_count[n];
				break;
			default:
				is_duplicate = true;
				break;
			}

		}

		if (!is_duplicate)
		{

			res.emplace_back(

					send_recv_s {

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

	return std::move(res);

}

void sync_update_dataset(DataSet * dset)
{
	std::vector<MPI_Request> requests = async_update_dataset(dset);

	MPI_Waitall(requests.size(), &(requests[0]), MPI_STATUSES_IGNORE);

}
std::vector<MPI_Request> async_update_dataset(DataSet * dset)
{
	return std::move(
			async_update_continue(decompose(dset->dataspace, dset->datatype),
					dset->data.get()));
}
std::vector<MPI_Request> async_update_continue(
		std::vector<send_recv_s> const & send_recv_list, void * data)

{
	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	static constexpr size_t ndims = 3;

	std::vector<MPI_Request> requests;

	requests.resize(send_recv_list.size() * 2);

	MPI_Request * req_it = &requests[0];

	for (auto const & item : send_recv_list)
	{

		MPI_Isend(data, 1, item.send_type.type(), item.remote, item.send_tag,
				mpi_comm.comm(), req_it);
		++req_it;
		MPI_Irecv(data, 1, item.recv_type.type(), item.remote, item.recv_tag,
				mpi_comm.comm(), req_it);
		++req_it;

	}

	return std::move(requests);
}

void sync_update_unordered(std::vector<send_recv_buffer_s> const & send_buffer,
		std::vector<send_recv_buffer_s> * recv_buffer)
{
	std::vector<MPI_Request> requests = async_update_unordered(send_buffer,
			recv_buffer);

	MPI_Waitall(requests.size(), &(requests[0]), MPI_STATUSES_IGNORE);

}

std::vector<MPI_Request> async_update_unordered(
		std::vector<send_recv_buffer_s> const & send_buffer,
		std::vector<send_recv_buffer_s> * recv_buffer)
{
	std::vector<MPI_Request> requests;

	MPIComm & global_comm = SingletonHolder<simpla::MPIComm>::instance();

	int num_of_reqs = send_buffer.size() + recv_buffer->size();

	requests.resize(num_of_reqs);

	MPI_Request * req_it = &requests[0];

	std::map<int, std::pair<size_t, std::shared_ptr<void> > > out_buffer;

	int count = 0;

	for (auto it = send_buffer.begin(), ie = send_buffer.end(); it != ie; ++it)
	{
		int remote, send_tag;
		size_t mem_size;
		std::shared_ptr<void> data;

		std::tie(remote, send_tag, mem_size, data) = *it;

		MPI_Isend(data.get(), mem_size, MPI_BYTE, remote, send_tag,
				global_comm.comm(), req_it);

		++req_it;

	}

	for (auto it = recv_buffer->begin(), ie = recv_buffer->end(); it != ie;
			++it)
	{
		int remote, recv_tag;

		std::tie(remote, recv_tag, std::ignore, std::ignore) = *it;

		MPI_Status status;

		MPI_Probe(remote, recv_tag, global_comm.comm(), &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int recv_mem_size = 0;

		MPI_Get_count(&status, MPI_BYTE, &recv_mem_size);

		if (recv_mem_size == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}

		std::shared_ptr<void> data;
		data = sp_alloc_memory(recv_mem_size);

		MPI_Irecv(data.get(), recv_mem_size,

		MPI_BYTE, remote, recv_tag, global_comm.comm(), req_it);

		std::get<2>(*it) = recv_mem_size;
		std::get<3>(*it) = data;
		++req_it;
	}

	return std::move(requests);

}

}
// namespace simpla
