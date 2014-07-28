/**
 * \file mpi_aux_functions.h
 *
 * \date    2014年7月18日  下午3:42:53 
 * \author salmon
 */

#ifndef MPI_AUX_FUNCTIONS_H_
#define MPI_AUX_FUNCTIONS_H_
extern "C"
{
#include <mpi.h>
}
#include "message_comm.h"
#include "../parallel/mpi_datatype.h"
namespace simpla
{

/**
 * @param pos in {0,count} out {begin,shape}
 */
template<typename Integral>
std::tuple<Integral, Integral> sync_global_location(Integral count)
{
	Integral begin = 0;

	if ( GLOBAL_COMM.is_ready() && GLOBAL_COMM.get_size() > 1)
	{

		auto communicator = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.get_size();
		int porcess_number = GLOBAL_COMM.get_rank();

		MPIDataType<Integral> m_type;

		std::vector<Integral> buffer;

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

template<typename T>
T * reduce(T * send_data, T * recv_data, size_t count, std::string const & op_c)
{

	MPIDataType<T> m_type;

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Reduce(reinterpret_cast<void*>(send_data), reinterpret_cast<void*>(recv_data), count, m_type.type(),
	        get_MPI_Op(op_c), 0, communicator);
	GLOBAL_COMM.barrier();

	return recv_data;

}

template<typename T>
T * allreduce(T * send_data, T * recv_data, size_t count, std::string const & op_c)
{

	MPIDataType<T> m_type;

	auto communicator = GLOBAL_COMM.comm();
	GLOBAL_COMM.barrier();
	MPI_Allreduce(reinterpret_cast<void*>(send_data), reinterpret_cast<void*>(recv_data), count, m_type.type(),
	        get_MPI_Op(op_c), communicator);
	GLOBAL_COMM.barrier();

	return recv_data;
}

template<typename T>
T reduce(T send, std::string const & op_c = "Sum")
{
	T recv;

	reduce(&send, &recv, 1, op_c);

	return recv;
}

template<typename T>
void reduce(T * p_send, std::string const & op_c = "Sum")
{
	T recv;

	reduce(p_send, &recv, 1, op_c);

	*p_send = recv;

}

template<int DIMS, typename T>
nTuple<DIMS, T> reduce(nTuple<DIMS, T> const & send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	reduce(&send[0], &recv[0], DIMS, op_c);

	return recv;
}

template<int DIMS, typename T>
void reduce(nTuple<DIMS, T> * p_send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	reduce(&(*p_send)[0], &recv[0], DIMS, op_c);

	*p_send = recv;

}

template<typename T>
T allreduce(T send, std::string const & op_c = "Sum")
{
	T recv;

	allreduce(&send, &recv, 1, op_c);

	return recv;
}

template<typename T>
void allreduce(T * p_send, std::string const & op_c = "Sum")
{
	T recv;

	allreduce(p_send, &recv, 1, op_c);

	*p_send = recv;

}

template<int DIMS, typename T>
nTuple<DIMS, T> allreduce(nTuple<DIMS, T> const & send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	allreduce(&send[0], &recv[0], DIMS, op_c);

	return recv;
}

template<int DIMS, typename T>
void allreduce(nTuple<DIMS, T> * p_send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	allreduce(&(*p_send)[0], &recv[0], DIMS, op_c);

	*p_send = recv;

}

struct MPI_data_pack_s
{

	std::shared_ptr<ByteType> buffer;
	int count;
	DataType data_type;
	int node_id;
	int tag;

};
void send_recv(std::vector<MPI_data_pack_s> & send_buffer, std::vector<MPI_data_pack_s> & recv_buffer);

}  // namespace simpla
#endif /* MPI_AUX_FUNCTIONS_H_ */
