/*
 * sync_array.h
 *
 *  Created on: 2014年5月26日
 *      Author: salmon
 */

#ifndef SYNC_ARRAY_H_
#define SYNC_ARRAY_H_

#include <mpi.h>
#include <stddef.h>
#include <memory>

#include "../utilities/memory_pool.h"
#include "../utilities/singleton_holder.h"
#include "message_comm.h"
#include "mpi_datatype.h"

namespace simpla
{

template<typename TR, typename TF>
void SyncGhost(int dest, TR out, TR in, TF *data)
{
	typedef decltype((*data)[out.begin()]) value_type;

	auto & field = *data;

	std::shared_ptr<value_type> buff_out = MEMPOOL.allocate_shared_ptr < value_type > (out.size());
	std::shared_ptr<value_type> buff_in = MEMPOOL.allocate_shared_ptr < value_type > (in.size());

	size_t count = 0;
	for (auto it : out)
	{
		*(buff_out + count) = field[it];
	}
	MPI_Datatype data_type = MPIDataType<value_type>().type();

	int out_tag = GLOBAL_COMM.GetRank();
	int in_tag = dest;

	MPI_Sendrecv(

	buff_out.get(), out.size(), data_type, dest, out_tag,

	buff_in.get(), in.size(), data_type, dest, in_tag,

	GLOBAL_COMM.GetComm(),MPI_STATUS_IGNORE);

	count = 0;
	for (auto it : in)
	{
		field[it] = *(buff_in + count);
	}
);
}

}  // namespace simpla

#endif /* SYNC_ARRAY_H_ */
