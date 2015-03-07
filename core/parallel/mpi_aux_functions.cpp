/**
 * \file mpi_aux_functions.cpp
 *
 * \date    2014年7月29日  上午8:16:09 
 * \author salmon
 */

#include "mpi_aux_functions.h"

#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../gtl/primitives.h"
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"

extern "C"
{
#include <mpi.h>
}

#include "mpi_comm.h"
#include "mpi_datatype.h"
#include "distributed_array.h"

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
		MPI_Bcast(&count, 1, m_type.type(), 0, comm);
	}

	return std::make_tuple(begin, count);

}
inline MPI_Op get_MPI_Op(std::string const & op_c)
{
	MPI_Op op = MPI_SUM;

	if (op_c == "Max")
	{
		op = MPI_MAX;
	}
	else if (op_c == "Min")
	{
		op = MPI_MIN;
	}
	else if (op_c == "Sum")
	{
		op = MPI_SUM;
	}
	else if (op_c == "Prod")
	{
		op = MPI_PROD;
	}
	else if (op_c == "LAND")
	{
		op = MPI_LAND;
	}
	else if (op_c == "LOR")
	{
		op = MPI_LOR;
	}
	else if (op_c == "BAND")
	{
		op = MPI_BAND;
	}
	else if (op_c == "Sum")
	{
		op = MPI_BOR;
	}
	else if (op_c == "Sum")
	{
		op = MPI_MAXLOC;
	}
	else if (op_c == "Sum")
	{
		op = MPI_MINLOC;
	}
	return op;
}

void reduce(void const* send_data, void * recv_data, size_t count,
		DataType const & data_type, std::string const & op_c)
{
	auto m_type = MPIDataType::create(data_type);

	auto comm = GLOBAL_COMM.comm();
	MPI_Barrier(comm);
	MPI_Reduce(const_cast<void*>(send_data), (recv_data), count, m_type.type(),
			get_MPI_Op(op_c), 0, comm);
	MPI_Barrier(comm);

}

void allreduce(void const* send_data, void * recv_data, size_t count,
		DataType const & data_type, std::string const & op_c)
{

	auto m_type = MPIDataType::create(data_type);

	auto comm = GLOBAL_COMM.comm();
	MPI_Barrier(comm);
	MPI_Allreduce(const_cast<void*>(send_data),
			reinterpret_cast<void*>(recv_data), count, m_type.type(),
			get_MPI_Op(op_c), comm);
	MPI_Barrier(comm);

}

std::tuple<std::shared_ptr<ByteType>, int> update_ghost_unorder(
		void const* send_buffer, std::vector<

		std::tuple<int, // dest;
				int, // send_tag;
				int, // recv_tag;
				int, // send buffer begin;
				int  // send buffer size;
				>> const & info)
{
	auto comm = GLOBAL_COMM.comm();
	MPI_Barrier(comm);

	MPI_Request requests[info.size() * 2];

	int req_count = 0;

	// send
	for (auto const & item : info)
	{

		MPI_Isend( reinterpret_cast<ByteType*>(const_cast<void* >(send_buffer))+std::get<3>(item) ,
				std::get<4>(item), MPI_BYTE, std::get<0>(item), std::get<1>(item),
				GLOBAL_COMM.comm(), &requests[req_count]);

		++req_count;
	}

	std::vector<int> mem_size;

	for (auto const & item : info)
	{
		MPI_Status status;

		MPI_Probe(std::get<0>(item), std::get<2>(item), GLOBAL_COMM.comm(), &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int tmp = 0;
		MPI_Get_count(&status, MPI_BYTE, &tmp);

		if (tmp == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}
		else
		{
			mem_size.push_back(tmp);
		}
	}
	int recv_buffer_size=0;
	for(auto const & v:mem_size)
	{	recv_buffer_size+=v;}

	auto recv_buffer = sp_make_shared_array<ByteType>(recv_buffer_size);

	int pos = 0;
	for (int i = 0; i < info.size(); ++i)
	{

		MPI_Irecv(recv_buffer.get() + pos, mem_size[i], MPI_BYTE, std::get<0>(info[i]), std::get<2>(info[i]),
				GLOBAL_COMM.comm(), &requests[req_count] );

		pos+= mem_size[i];
		++req_count;
	}

	MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
	MPI_Barrier(comm);

	return std::make_tuple(recv_buffer,recv_buffer_size);
}

void bcast_string(std::string * filename_)
{

	if (!GLOBAL_COMM.is_valid()) return;

	int name_len;

	if (GLOBAL_COMM.process_num()==0) name_len=filename_->size();

	MPI_Bcast(&name_len, 1, MPI_INT, 0, GLOBAL_COMM.comm());

	std::vector<char> buffer(name_len);

	if (GLOBAL_COMM.process_num()==0)
	{
		std::copy(filename_->begin(),filename_->end(),buffer.begin());
	}

	MPI_Bcast((&buffer[0]), name_len, MPI_CHAR, 0, GLOBAL_COMM.comm());

	buffer.push_back('\0');

	if (GLOBAL_COMM.process_num()!=0)
	{
		*filename_=&buffer[0];
	}

}

void get_ghost_shape(size_t ndims, size_t const * l_dims,
		size_t const * l_offset, size_t const * l_stride,
		size_t const * l_count, size_t const * l_block,
		size_t const * ghost_width,
		std::vector<mpi_ghosts_shape_s>* send_recv_list)
{

	nTuple<size_t, MAX_NDIMS_OF_ARRAY> send_count, send_offset;
	nTuple<size_t, MAX_NDIMS_OF_ARRAY> recv_count, recv_offset;

	for (unsigned int tag = 0, tag_e = (1UL << (ndims * 2)); tag < tag_e; ++tag)
	{
		nTuple<int, 3> coords_shift;

		bool tag_is_valid = false;

		for (int n = 0; n < ndims; ++n)
		{
			coords_shift[n] = ((tag >> (n * 2)) & 3UL) - 1;

			switch (coords_shift[n])
			{
			case 0:
				send_count[n] = l_count[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = l_count[n];
				recv_offset[n] = l_offset[n];
				break;
			case -1:

				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n];
				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] - ghost_width[n];
				tag_is_valid = true;
				break;

			case 1:

				send_count[n] = ghost_width[n];
				send_offset[n] = l_offset[n] + l_count[n] - ghost_width[n];
				recv_count[n] = ghost_width[n];
				recv_offset[n] = l_offset[n] + l_count[n];
				tag_is_valid = true;
				break;
			}

			if (send_count[n] == 0 || recv_count[n] == 0)
			{
				tag_is_valid = false;
				break;
			}
		}

		if (tag_is_valid)
		{

			send_recv_list->emplace_back(mpi_ghosts_shape_s
			{ coords_shift, send_offset, send_count, recv_offset, recv_count }

			);
		}

	}
}
}
// namespace simpla
