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

/**
 * @param   in count out {begin,total}
 */

std::tuple<int, int> sync_global_location(int count);

template<typename Integral>
std::tuple<Integral, Integral> sync_global_location(Integral count)
{
	int n = count;
	int begin, total;
	std::tie(begin, total) = sync_global_location(n);
	Integral rbegin = begin;
	Integral rtotal = static_cast<Integral>(total);

	return std::make_tuple(rbegin, rtotal);

}
void reduce(void const* send_data, void * recv_data, size_t count,
		DataType const & data_type, std::string const & op_c);

void allreduce(void const* send_data, void * recv_data, size_t count,
		DataType const & data_type, std::string const & op_c);

template<typename T>
void reduce(T * send_data, T * recv_data, size_t count,
		std::string const & op_c = "Sum")
{
	reduce(send_data, recv_data, count, DataType::create<T>(), op_c);

}

template<typename T>
void allreduce(T * send_data, T * recv_data, size_t count,
		std::string const & op_c = "Sum")
{
	allreduce(send_data, recv_data, count, DataType::create<T>(), op_c);
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

template<typename T, int DIMS>
nTuple<T, DIMS> reduce(nTuple<T, DIMS> const & send, std::string const & op_c =
		"Sum")
{
	nTuple<T, DIMS> recv;

	reduce(&send[0], &recv[0], DIMS, op_c);

	return recv;
}

template<typename T, int DIMS>
void reduce(nTuple<T, DIMS> * p_send, std::string const & op_c = "Sum")
{
	nTuple<T, DIMS> recv;

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

template<typename T, int DIMS>
nTuple<T, DIMS> allreduce(nTuple<T, DIMS> const & send,
		std::string const & op_c = "Sum")
{
	nTuple<T, DIMS> recv;

	allreduce(&send[0], &recv[0], DIMS, op_c);

	return recv;
}

template<typename T, int DIMS>
void allreduce(nTuple<T, DIMS> * p_send, std::string const & op_c = "Sum")
{
	nTuple<T, DIMS> recv;

	allreduce(&(*p_send)[0], &recv[0], DIMS, op_c);

	*p_send = recv;

}
std::tuple<std::shared_ptr<ByteType>, int> update_ghost_unorder(
		void const* send_buffer, std::vector<std::tuple<int, // dest;
				int, // send_tag;
				int, // recv_tag;
				int, // send buffer begin;
				int  // send buffer size;
				>> const & info);
}  // namespace simpla
#endif /* MPI_AUX_FUNCTIONS_H_ */
