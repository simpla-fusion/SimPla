/**
 * \file mpi_aux_functions.h
 *
 * \date    2014年7月18日  下午3:42:53 
 * \author salmon
 */

#ifndef MPI_AUX_FUNCTIONS_H_
#define MPI_AUX_FUNCTIONS_H_

#include "../utilities/data_type.h"

namespace simpla
{
void reduce(void const* send_data, void * recv_data, size_t count, DataType const & data_type,
        std::string const & op_c);

void allreduce(void const* send_data, void * recv_data, size_t count, DataType const & data_type,
        std::string const & op_c);
template<typename T>
T reduce(T send, std::string const & op_c = "Sum")
{
	T recv;

	reduce(&send, &recv, 1, DataType::create<T>(), op_c);

	return recv;
}

template<typename T>
void reduce(T * p_send, std::string const & op_c = "Sum")
{
	T recv;

	reduce(p_send, &recv, 1, DataType::create<T>(), op_c);

	*p_send = recv;

}

template<int DIMS, typename T>
nTuple<DIMS, T> reduce(nTuple<DIMS, T> const & send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	reduce(&send[0], &recv[0], DIMS, DataType::create<T>(), op_c);

	return recv;
}

template<int DIMS, typename T>
void reduce(nTuple<DIMS, T> * p_send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	reduce(&(*p_send)[0], &recv[0], DIMS, DataType::create<T>(), op_c);

	*p_send = recv;

}

template<typename T>
T allreduce(T send, std::string const & op_c = "Sum")
{
	T recv;

	allreduce(&send, &recv, 1, DataType::create<T>(), op_c);

	return recv;
}

template<typename T>
void allreduce(T * p_send, std::string const & op_c = "Sum")
{
	T recv;

	allreduce(p_send, &recv, 1, DataType::create<T>(), op_c);

	*p_send = recv;

}

template<int DIMS, typename T>
nTuple<DIMS, T> allreduce(nTuple<DIMS, T> const & send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	allreduce(&send[0], &recv[0], DIMS, DataType::create<T>(), op_c);

	return recv;
}

template<int DIMS, typename T>
void allreduce(nTuple<DIMS, T> * p_send, std::string const & op_c = "Sum")
{
	nTuple<DIMS, T> recv;

	allreduce(&(*p_send)[0], &recv[0], DIMS, DataType::create<T>(), op_c);

	*p_send = recv;

}
class DistributedArray;

void update_ghosts(void* data, DataType const & data_desc, DistributedArray const & global_array);

template<typename TV>
void update_ghosts(TV* data, DistributedArray const & global_array)
{
	update_ghosts(reinterpret_cast<void *>(data), DataType::create<TV>(), global_array);
}

/**
 *
 * @param send_buffer
 * @return tuple(recv_buffer, buffer size)
 */
std::tuple<std::shared_ptr<ByteType>, int> update_ghosts_unorder(void const * send_buffer,

std::tuple<int, // dest;
        int, // send_tag;
        int, // recv_tag;
        int, // send buffer begin;
        int  // send buffer size;
        > const & info);
}  // namespace simpla
#endif /* MPI_AUX_FUNCTIONS_H_ */
