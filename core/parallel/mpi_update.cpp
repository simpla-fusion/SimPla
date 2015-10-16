/**
 * @file mpi_update.cpp
 *
 * @date    2014-7-29  AM8:32:26
 * @author salmon
 */

#include "mpi_update.h"
#include "mpi_datatype.h"

#include "../dataset/dataset.h"
#include "../gtl/utilities/log.h"

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
		int process_num = GLOBAL_COMM.process_num( );

		MPIDataType m_type =MPIDataType::create<int>();

		std::vector<int> buffer;

		if (process_num == 0)
		buffer.resize(num_of_process);
		MPI_Barrier(comm);

		MPI_Gather(&count, 1, m_type.type(), &buffer[0], 1, m_type.type(), 0, comm);

		MPI_Barrier(comm);

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
		MPI_Barrier(comm);
		MPI_Scatter(&buffer[0], 1, m_type.type(), &begin, 1, m_type.type(), 0, comm);
		MPI_Barrier(comm);
		MPI_Bcast(&count, 1, m_type.type(), 0, comm);
		MPI_Barrier(comm);

	}

	return std::make_tuple(begin, count);

}

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
