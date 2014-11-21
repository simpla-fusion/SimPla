/**
 * \file mpi_aux_functions.cpp
 *
 * \date    2014年7月29日  上午8:16:09 
 * \author salmon
 */

#include "mpi_aux_functions.h"

extern "C"
{
#include <mpi.h>
}

#include "mpi_comm.h"
#include "mpi_datatype.h"
#include "distributed_array.h"

#include "../utilities/log.h"
#include "../utilities/memory_pool.h"
namespace simpla
{

/**
 * @param pos in {0,count} out {begin,shape}
 */
std::tuple<int, int> sync_global_location(int count)
{
	int begin = 0;

	if ( GLOBAL_COMM.is_valid() && GLOBAL_COMM.get_size() > 1)
	{

		auto communicator = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.get_size();
		int porcess_number = GLOBAL_COMM.get_rank();

		MPIDataType m_type =MPIDataType::create<int>();

		std::vector<int> buffer;

		if (porcess_number == 0)
		buffer.resize(num_of_process);

		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, communicator);

		MPI_Barrier(communicator);

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
		MPI_Barrier(communicator);
		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, communicator);
		MPI_Bcast(&count, 1, m_type.type(), 0, communicator);
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

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Reduce(const_cast<void*>(send_data), (recv_data), count, m_type.type(),
			get_MPI_Op(op_c), 0, communicator);
	GLOBAL_COMM.barrier();

}

void allreduce(void const* send_data, void * recv_data, size_t count,
		DataType const & data_type, std::string const & op_c)
{

	auto m_type = MPIDataType::create(data_type);

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Allreduce(const_cast<void*>(send_data),
			reinterpret_cast<void*>(recv_data), count, m_type.type(),
			get_MPI_Op(op_c), communicator);
	GLOBAL_COMM.barrier();

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
	GLOBAL_COMM.barrier();

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
	int recv_buffer_size=std::accumulate(mem_size.begin(),mem_size.end(),0);
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
	GLOBAL_COMM.barrier();

	return std::make_tuple(recv_buffer,recv_buffer_size);
}
}
		// namespace simpla
