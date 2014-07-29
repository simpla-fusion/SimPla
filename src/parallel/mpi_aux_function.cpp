/**
 * \file mpi_aux_function.cpp
 *
 * \date    2014年7月28日  上午11:11:49 
 * \author salmon
 */

#include "mpi_aux_functions.h"
#include "../utilities/log.h"
#include "../utilities/memory_pool.h"

namespace simpla
{
void send_recv(std::vector<MPI_data_pack_s> const& send_buffer, std::vector<MPI_data_pack_s> & recv_buffer)
{
	MPI_Request requests[send_buffer.size() + recv_buffer.size()];

	int req_count = 0;

	for (auto const & item : recv_buffer)
	{
		MPI_Status status;

		MPI_Probe(item.node_id, item.tag, GLOBAL_COMM.comm(), &status);

		// When probe returns, the status object has the size and other
		// attributes of the incoming message. Get the size of the message
		int mem_size = 0;

		MPI_Get_count(&status, MPI_BYTE, &mem_size);

		if (mem_size == MPI_UNDEFINED)
		{
			RUNTIME_ERROR("Update Ghosts Particle fail");
		}
		item.buffer = MEMPOOL.allocate_byte_shared_ptr(mem_size);

		item.count = mem_size / item.data_type.size_in_byte();

		MPI_Irecv(item.buffer.get(), mem_size, MPI_BYTE, item.node_id, item.tag,
		GLOBAL_COMM.comm(),
		&requests[req_count]

		);

		++req_count;
	}

	// send
	for (auto const & item : send_buffer)
	{

		MPI_Isend(item.buffer.get(), item.count * item.data_type.size_in_byte(),
		MPI_BYTE, item.node_id, item.tag, GLOBAL_COMM.comm(), &requests[req_count]);

		++req_count;
	}

	MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

}

}  // namespace simpla
