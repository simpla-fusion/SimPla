/*
 * sync_array.h
 *
 *  created on: 2014-5-26
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
template<unsigned int N>
struct DistributedArray
{
public:
	static constexpr  unsigned int  NDIMS = N;

	DistributedArray()

	{
	}

	template<typename ...Args>
	DistributedArray(Args && ... args)

	{
		init(std::forward<Args >(args)...);
	}

	~DistributedArray()
	{

	}

	void init(nTuple<NDIMS, size_t> const &global_start, nTuple<NDIMS, size_t> const &global_count,  unsigned int  num_process,
	         unsigned int  process_num, size_t gw)

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
				if (local_.outer_begin[i] < global_start[i]
				        && node.remote.inner_start[i] > node.remote.outer_start[i] + global_count[i])
				{
					node.remote.outer_start[i] -= global_count[i];
					node.remote.inner_start[i] -= global_count[i];
				}

				if (local_.outer_begin[i] + local_.outer_end[i] > global_start[i] + global_count[i]
				        && node.remote.inner_start[i] + node.remote.inner_count[i]
				                < local_.outer_begin[i] - global_count[i])
				{
					node.remote.outer_start[i] += global_count[i];
					node.remote.inner_start[i] += global_count[i];
				}
				size_t start, end;

				start = std::max(local_.outer_begin[i], node.remote.inner_start[i]);
				end = std::min(local_.inner_begin[i], node.remote.inner_start[i] + node.remote.inner_count[i]);

				if (end > start)
				{
					node.remote.inner_count[i] = end - start;
					node.remote.inner_start[i] = start;
					need_update = true;
					continue;
				}

				start = std::max(local_.inner_begin[i] + local_.inner_end[i], node.remote.inner_start[i]);
				end = std::min(local_.outer_begin[i] + local_.outer_end[i],
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
	void updateGhost(TV* data, MPI_Comm comm = nullptr) const
	{
		if (comm == nullptr)
		{
			comm = GLOBAL_COMM.GetComm();
		}
		MPI_Win win;

		MPI_Win_create(data, NProduct(local_.outer_end), sizeof(TV), MPI_INFO_NULL, comm, &win);

		for (auto const & neighbour : recv_)
		{

			MPI_Get(

					data, 1,

					MPIDataType<TV>(comm, local_.outer_end, neighbour.remote.inner_count,
							neighbour.local_start - local_.outer_begin).type(),

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
		                nTuple<NDIMS, size_t> outer_begin;
		                nTuple<NDIMS, size_t> outer_end;
		                nTuple<NDIMS, size_t> inner_begin;
		                nTuple<NDIMS, size_t> inner_end;
	                };
	                struct neighbour_s
	                {
		                  unsigned int   dest;

		                sub_array_s remote;

		                nTuple<NDIMS, size_t> local_start;
	                };

	                sub_array_s local_;

	                  unsigned int   array_order_ = MPI_ORDER_C;

	                std::vector<neighbour_s> recv_;
	                void Decompose(int num_process,  unsigned int  process_num, size_t gw, nTuple<NDIMS, size_t> const &global_start,
			nTuple<NDIMS, size_t> const &global_count, sub_array_s * local) const
	{

	}

}
;

}
// namespace simpla

#endif /* SYNC_ARRAY_H_ */
