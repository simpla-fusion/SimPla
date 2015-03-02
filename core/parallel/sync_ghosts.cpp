/**
 * @file sync_ghosts.cpp
 *
 * @date    2014年7月29日  上午8:32:26
 * @author salmon
 */

#include "../utilities/log.h"
#include "../dataset/dataset.h"
#include "mpi_comm.h"
#include "mpi_datatype.h"
namespace simpla
{
struct send_recv_s
{
	int remote;
	int send_tag;
	int recv_tag;
	nTuple<size_t, 3> dimensions;
	nTuple<size_t, 3> send_offset;
	nTuple<size_t, 3> send_count;
	nTuple<size_t, 3> recv_offset;
	nTuple<size_t, 3> recv_count;
};
std::vector<send_recv_s> decompose(DataSpace const & dataspace, size_t flag)
{
	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	static constexpr size_t ndims = 3;

	std::vector<send_recv_s> res;

	auto ghost_width = dataspace.ghost_width();

	nTuple<size_t, ndims> g_dims, g_offset, g_count;
	std::tie(ndims, g_dims, g_offset, g_count, std::ignore, std::ignore) =
			dataspace.global().shape();

	nTuple<size_t, ndims> l_dims, l_offset, l_count;
	std::tie(ndims, l_dims, l_offset, l_count, std::ignore, std::ignore) =
			dataspace.local().shape();

	nTuple<size_t, ndims> send_count, send_offset;
	nTuple<size_t, ndims> recv_count, recv_offset;

	static constexpr size_t ndims = 3;

	auto ghost_width = dataspace.ghost_width();

	nTuple<size_t, 3> g_dims, g_offset, g_count;
	std::tie(ndims, g_dims, g_offset, g_count, std::ignore, std::ignore) =
			dataspace.global().shape();

	nTuple<size_t, 3> l_dims, l_offset, l_count;
	std::tie(ndims, l_dims, l_offset, l_count, std::ignore, std::ignore) =
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

		if (is_duplicate)
			continue;

		auto remote = mpi_comm.get_neighbour(coords_shift);

		res.emplace_back(remote,

		static_cast<int>(std::hash(send_offset)),

		static_cast<int>(std::hash(send_offset)),

		send_offset,

		send_count,

		recv_offset,

		recv_count);

	}

	return std::move(res);
}

void sync_ghosts(DataSet * data, size_t flag)
{
	sync_ghosts(decompose(data->dataspace, flag), data->datatype,
			data->data.get(), flag);
}
void sync_ghosts(std::vector<send_recv_s> const & send_recv_list,
		DataType const & datatype, void * data, size_t flag)

{
	auto & mpi_comm = SingletonHolder<simpla::MPIComm>::instance();

	static constexpr size_t ndims = 3;

	MPI_Request request[send_recv_list.size() * 2];

	MPI_Request * req_it = &request[0];

	for (auto const & item : send_recv_list)
	{

		MPIDataType send_type = MPIDataType::create(datatype, ndims,
				&item.dimensions[0], &item.send_count[0], &item.send_offset[0]);

		MPIDataType recv_type = MPIDataType::create(datatype, ndims,
				&item.dimensions[0], &item.recv_count[0], &item.recv_offset[0]);

		MPI_Isend(data, 1, send_type.type(), item.remote, item.send_tag,
				mpi_comm.comm(), req_it);
		++req_it;
		MPI_Irecv(data, 1, recv_type.type(), item.remote, item.recv_tag,
				mpi_comm.comm(), req_it);
		++req_it;

	}

	MPI_Waitall(send_recv_list.size() * 2, request, MPI_STATUSES_IGNORE);

}

void sync_ghosts_unordered(DataSpace const & space,
		std::map<int, std::pair<size_t, std::shared_ptr<void> > > const & send_buffer,
		std::map<int, std::pair<size_t, std::shared_ptr<void> > > & recv_buffer,
		size_t flag)
{
	sync_ghosts_unordered(decompose(space, flag), send_buffer, recv_buffer,
			flag);
}

void sync_ghosts_unordered(std::vector<send_recv_s> const & send_recv_list,
		std::map<int, std::pair<size_t, std::shared_ptr<void> > > const & send_buffer,
		std::map<int, std::pair<size_t, std::shared_ptr<void> > > & recv_buffer,
		size_t flag)
{
	MPIComm & global_comm = SingletonHolder<simpla::MPIComm>::instance();

	MPI_Request request[send_recv_list.size() * 2];

	MPI_Request * req_it = &request[0];

	std::map<int, std::pair<size_t, std::shared_ptr<void> > > out_buffer;

	int count = 0;

	for (auto const & item : send_recv_list)
	{
		auto it = send_buffer.find(item.remote);
		if (it != send_buffer.end())
		{
			MPI_Isend(it->second.second.get(), it->second.first,
			MPI_BYTE, item.remote, item.send_tag, global_comm.comm(), req_it);
		}
		else
		{
			MPI_Isend(nullptr, 0,
			MPI_BYTE, item.remote, item.send_tag, global_comm.comm(), req_it);
		}
		++req_it;

	}

	for (auto const & item : send_recv_list)
	{

		MPI_Status status;

		MPI_Probe(item.remote, item.recv_tag, global_comm.comm(), &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int recv_mem_size = 0;

		MPI_Get_count(&status, MPI_BYTE, &recv_mem_size);

		if (recv_mem_size == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}
		recv_buffer[count].second = sp_make_shared_array<ByteType>(
				recv_mem_size / sizeof(ByteType));

		MPI_Irecv(recv_buffer[count].second.get(), recv_mem_size,
		MPI_BYTE, item.remote, item.recv_tag, global_comm.comm(), req_it);
		++req_it;
	}

	MPI_Waitall(send_recv_list.size() * 2, request, MPI_STATUSES_IGNORE);
}

//void decomposer_(size_t num_process, size_t process_num, size_t gw,
//		size_t ndims, size_t const *global_begin, size_t const * global_end,
//		size_t * local_outer_begin, size_t * local_outer_end,
//		size_t * local_inner_begin, size_t * local_inner_end)
//{
//	local_outer_end = global_end;
//	local_outer_begin = global_begin;
//	local_inner_end = global_end;
//	local_inner_begin = global_begin;
//
//	if (num_process <= 1)
//		return;
//
//	int n = 0;
//	long L = 0;
//	for (int i = 0; i < ndims; ++i)
//	{
//		if ((global_end[i] - global_begin[i]) > L)
//		{
//			L = (global_end[i] - global_begin[i]);
//			n = i;
//		}
//	}
//
//	if ((2 * gw * num_process > (global_end[n] - global_begin[n])
//			|| num_process > (global_end[n] - global_begin[n])))
//	{
//
//		RUNTIME_ERROR("Array is too small to split");
//
////		if (process_num > 0)
////			local_outer_end = local_outer_begin;
//	}
//	else
//	{
//		local_inner_begin[n] = ((global_end[n] - global_begin[n]) * process_num)
//				/ num_process + global_begin[n];
//		local_inner_end[n] = ((global_end[n] - global_begin[n])
//				* (process_num + 1)) / num_process + global_begin[n];
//		local_outer_begin[n] = local_inner_begin[n] - gw;
//		local_outer_end[n] = local_inner_end[n] + gw;
//	}
//
//}
//
//void DistributedArray::Decompose(size_t gw)
//{
//	int num_process = GLOBAL_COMM.get_size();
//	unsigned int process_num = GLOBAL_COMM.get_rank();
//
//	decomposer_(num_process, process_num, gw, ndims,  //
//			global_begin_, global_end_,//
//			local_.outer_begin, local_.outer_end,//
//			local_.inner_begin, local_.inner_end);
//
//	self_id_ = (process_num);
//
//	if (num_process <= 1)
//	return;
//
//	global_strides_[0] = 1;
//
//	for (int i = 1; i < ndims; ++i)
//	{
//		global_strides_[i] = (global_end_[i] - global_begin_[i]) * global_strides_[i - 1];
//	}
//
//	for (int dest = 0; dest < num_process; ++dest)
//	{
//		if (dest == self_id_)
//		continue;
//
//		sub_array_s node;
//
//		decomposer_(num_process, dest, gw, ndims, global_begin_, global_end_, node.outer_begin, node.outer_end,
//				node.inner_begin, node.inner_end);
//
//		sub_array_s remote;
//
//		for (unsigned long s = 0, s_e = (1UL << (ndims * 2)); s < s_e; ++s)
//		{
//			remote = node;
//
//			bool is_duplicate = false;
//
//			for (int i = 0; i < ndims; ++i)
//			{
//
//				int n = (s >> (i * 2)) & 3UL;
//
//				if (n == 3)
//				{
//					is_duplicate = true;
//					continue;
//				}
//
//				auto L = (global_end_[i] - global_begin_[i]) * ((n + 1) % 3 - 1);
//
//				remote.outer_begin[i] += L;
//				remote.outer_end[i] += L;
//				remote.inner_begin[i] += L;
//				remote.inner_end[i] += L;
//
//			}
//			if (!is_duplicate)
//			{
//				bool f_inner = Clipping(ndims, local_.outer_begin, local_.outer_end, remote.inner_begin,
//						remote.inner_end);
//				bool f_outer = Clipping(ndims, local_.inner_begin, local_.inner_end, remote.outer_begin,
//						remote.outer_end);
//
//				bool flag = f_inner && f_outer;
//
//				for (int i = 0; i < ndims; ++i)
//				{
//					flag = flag && (remote.outer_begin[i] != remote.outer_end[i]);
//				}
//				if (flag)
//				{
//					send_recv_.emplace_back(send_recv_s(
//									{	dest, hash(remote.outer_begin), hash(remote.inner_begin),
//										remote.outer_begin, remote.outer_end, remote.inner_begin, remote.inner_end}));
//				}
//			}
//		}
//	}
//
//}

}
// namespace simpla
