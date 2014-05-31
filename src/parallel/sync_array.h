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
#include <map>
#include <vector>
#include <pair>
#include "mpi_datatype.h"

namespace simpla
{
template<int N>
struct DistributedArray
{
public:
	static constexpr int NDIMS = N;

	DistributedArray()

	{
	}

	template<typename ...Args>
	DistributedArray(Args const & ... args)

	{
		Init(std::forward<Args const &>(args)...);
	}

	~DistributedArray()
	{

	}

	void Init(nTuple<NDIMS, size_t> const &global_start, nTuple<NDIMS, size_t> const &global_count, int num_process,
	        int process_num, size_t gw)

	{
		Decompose(num_process, process_num, gw, global_start, global_count, &local_);

		for (int n = 0; n < num_process; ++n)
		{

			bool need_update = false;

			neighbour_s node;

			node.dest = n;

			Decompose(num_process, n, gw, global_start, global_count, &(node.remote));

			for (int i = 0; i < NDIMS; ++i)
			{
				// periodic shift
				if (local_.outer_start[i] < global_start[i]
				        && node.remote.inner_start[i] > node.remote.outer_start[i] + global_count[i])
				{
					node.remote.outer_start[i] -= global_count[i];
					node.remote.inner_start[i] -= global_count[i];
				}

				if (local_.outer_start[i] + local_.outer_count[i] > global_start[i] + global_count[i]
				        && node.remote.inner_start[i] + node.remote.inner_count[i]
				                < local_.outer_start[i] - global_count[i])
				{
					node.remote.outer_start[i] += global_count[i];
					node.remote.inner_start[i] += global_count[i];
				}
				size_t start, end;

				start = std::max(local_.outer_start[i], node.remote.inner_start[i]);
				end = std::min(local_.inner_start[i], node.remote.inner_start[i] + node.remote.inner_count[i]);

				if (end > start)
				{
					node.remote.inner_count[i] = end - start;
					node.remote.inner_start[i] = start;
					need_update = true;
					continue;
				}

				start = std::max(local_.inner_start[i] + local_.inner_count[i], node.remote.inner_start[i]);
				end = std::min(local_.outer_start[i] + local_.outer_count[i],
				        node.remote.inner_start[i] + node.remote.inner_count[i]);

				if (end > start)
				{
					node.remote.inner_count[i] = end - start;
					node.remote.inner_start[i] = start;
					need_update = true;
					continue;
				}
			}

			if (need_update)
			{
				recv_.push_back(node);
			}

		}
	}
#ifdef USE_MPI
	template<typename TV>
	void UpdateGhost(TV* data, MPI_Comm comm = nullptr) const
	{
		if (comm == nullptr)
		{
			comm = GLOBAL_COMM.GetComm();
		}
		MPI_Win win;

		MPI_Win_create(data, NProduct(local_.outer_count), sizeof(TV), MPI_INFO_NULL, comm, &win);

		for (auto const & neighbour : recv_)
		{

			MPI_Get(

					data, 1,

					MPIDataType<TV>(comm, local_.outer_count, neighbour.remote.inner_count,
							neighbour.local_start - local_.outer_start).type(),

					neighbour.dest, 0, 1,

					MPIDataType<TV>(comm, neighbour.remote.outer_count, neighbour.remote.outer_count,
							neighbour.remote.inner_start - neighbour.remote.outer_start).type()

					, &win);

		}

		MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), win);

		MPI_Win_free(&win);

	}

#endif

                private:
	                struct sub_array_s
	                {
		                nTuple<NDIMS, size_t> outer_start;
		                nTuple<NDIMS, size_t> outer_count;
		                nTuple<NDIMS, size_t> inner_start;
		                nTuple<NDIMS, size_t> inner_count;
	                };
	                struct neighbour_s
	                {
		                unsigned int dest;

		                sub_array_s remote;

		                nTuple<NDIMS, size_t> local_start;
	                };

	                sub_array_s local_;

	                unsigned int array_order_ = MPI_ORDER_C;

	                std::vector<neighbour_s> recv_;
	                void Decompose(int num_process, int process_num, size_t gw, nTuple<NDIMS, size_t> const &global_start,
			nTuple<NDIMS, size_t> const &global_count, sub_array_s * local) const
	{

	}

}
;

}
// namespace simpla

#endif /* SYNC_ARRAY_H_ */
