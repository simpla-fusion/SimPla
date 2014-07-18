/**
 * \file mpi_aux_functions.h
 *
 * \date    2014年7月18日  下午3:42:53 
 * \author salmon
 */

#ifndef MPI_AUX_FUNCTIONS_H_
#define MPI_AUX_FUNCTIONS_H_

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

		auto comm = GLOBAL_COMM.comm();

		int num_of_process = GLOBAL_COMM.get_size();
		int porcess_number = GLOBAL_COMM.get_rank();

		MPIDataType<Integral> m_type;

		std::vector<Integral> buffer;

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
}  // namespace simpla
#endif /* MPI_AUX_FUNCTIONS_H_ */
