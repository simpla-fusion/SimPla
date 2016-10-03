/**
 * @file mpi_aux_functions.cpp
 *
 * @date    2014-7-29  AM8:16:09
 * @author salmon
 */

#include "MPIAuxFunctions.h"

#include <stddef.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../sp_def.h"
#include "../toolbox/Log.h"
#include "../toolbox/MemoryPool.h"

extern "C"
{
#include <mpi.h>
}

#include "MPIComm.h"
#include "MPIDataType.h"

namespace simpla { namespace parallel
{


inline MPI_Op get_MPI_Op(std::string const &op_c)
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

void reduce(void const *send_data, void *recv_data, size_t count,
            data_model::DataType const &data_type, std::string const &op_c)
{
    auto m_type = MPIDataType::create(data_type);

    auto comm = GLOBAL_COMM.comm();
    MPI_Barrier(comm);
    MPI_Reduce(const_cast<void *>(send_data), (recv_data), count, m_type.type(),
               get_MPI_Op(op_c), 0, comm);
    MPI_Barrier(comm);

}

void allreduce(void const *send_data, void *recv_data, size_t count,
               data_model::DataType const &data_type, std::string const &op_c)
{

    auto m_type = MPIDataType::create(data_type);

    auto comm = GLOBAL_COMM.comm();
    MPI_Barrier(comm);
    MPI_Allreduce(const_cast<void *>(send_data),
                  reinterpret_cast<void *>(recv_data), count, m_type.type(),
                  get_MPI_Op(op_c), comm);
    MPI_Barrier(comm);

}
//
//std::tuple<std::shared_ptr<byte_type>, int> update_ghost_unorder(
//		void const* m_send_links_, std::vector<
//
//		std::tuple<int, // dest;
//				int, // send_tag;
//				int, // recv_tag;
//				int, // send m_buffer begin;
//				int  // send m_buffer size;
//				>> const & info)
//{
//	auto comm = GLOBAL_COMM.comm();
//	MPI_Barrier(comm);
//
//	MPI_Request requests[info.size() * 2];
//
//	int req_count = 0;
//
//	// send
//	for (auto const & item : info)
//	{
//
//		MPI_Isend( reinterpret_cast<byte_type*>(const_cast<void* >(m_send_links_))+std::get<3>(item) ,
//				std::get<4>(item), MPI_BYTE, std::get<0>(item), std::get<1>(item),
//				GLOBAL_COMM.comm(), &requests[req_count]);
//
//		++req_count;
//	}
//
//	std::vector<int> mem_size;
//
//	for (auto const & item : info)
//	{
//		MPI_Status status;
//
//		MPI_Probe(std::get<0>(item), std::get<2>(item), GLOBAL_COMM.comm(), &status);
//
//		// When probe returns, the status object has the size and other
//		// attributes of the incoming message. Get the size of the message
//		int tmp = 0;
//		MPI_Get_count(&status, MPI_BYTE, &tmp);
//
//		if (tmp == MPI_UNDEFINED)
//		{
//			THROW_RUNTIME_ERROR("Update Ghosts Particle fail");
//		}
//		else
//		{
//			mem_size.push_back(tmp);
//		}
//	}
//	int recv_buffer_size=0;
//	for(auto const & v:mem_size)
//	{	recv_buffer_size+=v;}
//
//	auto m_recv_links_ = sp_make_shared_array<byte_type>(recv_buffer_size);
//
//	int pos = 0;
//	for (int i = 0; i < info.size(); ++i)
//	{
//
//		MPI_Irecv(m_recv_links_.get() + pos, mem_size[i], MPI_BYTE, std::get<0>(info[i]), std::get<2>(info[i]),
//				GLOBAL_COMM.comm(), &requests[req_count] );
//
//		pos+= mem_size[i];
//		++req_count;
//	}
//
//	MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
//	MPI_Barrier(comm);
//
//	return std::make_tuple(m_recv_links_,recv_buffer_size);
//}

void bcast_string(std::string *filename_)
{

    if (!GLOBAL_COMM.is_valid()) return;

    int name_len;

    if (GLOBAL_COMM.process_num() == 0) name_len = filename_->size();

    MPI_Bcast(&name_len, 1, MPI_INT, 0, GLOBAL_COMM.comm());

    std::vector<char> buffer(name_len);

    if (GLOBAL_COMM.process_num() == 0)
    {
        std::copy(filename_->begin(), filename_->end(), buffer.begin());
    }

    MPI_Bcast((&buffer[0]), name_len, MPI_CHAR, 0, GLOBAL_COMM.comm());

    buffer.push_back('\0');

    if (GLOBAL_COMM.process_num() != 0)
    {
        *filename_ = &buffer[0];
    }

}

}}//namespace simpla { namespace parallel

