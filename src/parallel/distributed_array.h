/*
 * distributed_array.h
 *
 *  Created on: 2014年5月30日
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_

#include <stddef.h>
#include <pair>
#include <vector>

#include "../fetl/ntuple.h"
#include "../utilities/singleton_holder.h"
#include "message_comm.h"

#ifdef USE_MPI
#	include <mpi.h>
#	include "mpi_datatype.h"
#endif

namespace simpla
{
template<int N>
struct DistributedArray
{
public:
	static constexpr int NDIMS = N;
	unsigned int array_order_ = MPI_ORDER_C;
private:
	struct sub_array_s
	{
		nTuple<NDIMS, size_t> outer_start;
		nTuple<NDIMS, size_t> outer_count;
		nTuple<NDIMS, size_t> inner_start;
		nTuple<NDIMS, size_t> inner_count;
	};
public:
	DistributedArray()
			: Decompose(DefaultDecomposer)
	{
	}

	template<typename ...Args>
	DistributedArray(Args const & ... args)
			: Decompose(DefaultDecomposer)
	{
		Init(std::forward<Args const &>(args)...);
	}

	~DistributedArray()
	{
	}

	void Init(int num_process, int process_num, size_t gw, nTuple<NDIMS, size_t> const &global_start,
	        nTuple<NDIMS, size_t> const &global_count);

	std::function<
	        void(int, int, unsigned int, nTuple<NDIMS, size_t> const &, nTuple<NDIMS, size_t> const &, sub_array_s *)> Decompose;

	template<typename TV> void UpdateGhost(TV* data) const;

#ifdef USE_MPI
	template<typename TV> void UpdateGhost(TV* data, MPI_Comm comm) const;
#endif

private:

	sub_array_s local_;

	std::map<int, sub_array_s> neighbours_;

	static void DefaultDecomposer(int num_process, int process_num, unsigned int gw,
	        nTuple<NDIMS, size_t> const &global_start, nTuple<NDIMS, size_t> const &global_count, sub_array_s *);

}
;

template<int N>
void DistributedArray<N>::DefaultDecomposer(int num_process, int process_num, unsigned int gw,
        nTuple<NDIMS, size_t> const &global_start, nTuple<NDIMS, size_t> const &global_count, sub_array_s * local)
{

}
template<int NDIMS>
bool Clipping(nTuple<NDIMS, size_t> const & l_start, nTuple<NDIMS, size_t> const &l_count,
        nTuple<NDIMS, size_t> *pr_start, nTuple<NDIMS, size_t> *pr_count)
{
	bool has_overlap = false;

	nTuple<NDIMS, size_t> & r_start = *pr_start;
	nTuple<NDIMS, size_t> & r_count = *pr_count;

	for (int i = 0; i < NDIMS; ++i)
	{

		size_t start = std::max(l_start[i], r_start[i]);
		size_t end = std::min(l_start[i] + l_count[i], r_start[i] + r_count[i]);

		if (end > start)
		{
			r_start[i] = start;
			r_count[i] = end - start;

			has_overlap = true;
			continue;
		}

	}
	return has_overlap;
}
template<int N>
void DistributedArray<N>::Init(int num_process, int process_num, size_t gw, nTuple<NDIMS, size_t> const &global_start,
        nTuple<NDIMS, size_t> const &global_count)

{
	Decompose(num_process, process_num, gw, global_start, global_count, &local_);

	for (int dest = 0; dest < num_process; ++dest)
	{

		sub_array_s node;

		Decompose(num_process, dest, gw, global_start, global_count, &(node));

		// assume no overlap in inner area
		// consider periodic boundary condition, traveral all neighbours

		sub_array_s remote;
		for (unsigned long s = 0, s_e = (1UL << (NDIMS * 2)); s < s_e; ++s)
		{
			remote = node;
			for (int i = 0; i < NDIMS; ++i)
			{
				int n = (s >> (i * 2)) & 3;
				if (n == 3)
					continue;

				n = (n + 1) % 3 - 1; // 0 1 2 => 0 1 -1

				remote.outer_start[i] += global_count[i] * n;
				remote.inner_start[i] += global_count[i] * n;
			}

			if (Clipping(local_.outer_start, local_.outer_count, &remote.inner_start, &remote.inner_count))
			{
				neighbours_[dest] = remote;
			}
		}

	}
}

#ifdef USE_MPI
template<int N>
template<typename TV>
void DistributedArray<N>::UpdateGhost(TV* data, MPI_Comm comm) const
{

	MPI_Win win;

	MPI_Win_create(data, NProduct(local_.outer_count), sizeof(TV), MPI_INFO_NULL, comm, &win);

	for (auto const & remote : neighbours_)
	{

		MPI_Get(

		data, 1,

		        MPIDataType<TV>(comm, local_.outer_count, remote.second.inner_count,
		                remote.second.inner_start - local_.outer_start).type(),

		        remote.first, 0, 1,

		        MPIDataType<TV>(comm, remote.second.outer_count, remote.second.inner_count,
		                remote.second.inner_start - remote.second.outer_start).type()

		        , &win);

	}

	MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), win);

	MPI_Win_free(&win);

}
#endif

template<int N>
template<typename TV>
void DistributedArray<N>::UpdateGhost(TV* data) const
{
#ifdef USE_MPI
	UpdateGhost(data, SingletonHolder<simpla::MessageComm>::instance().GetComm());
#else
	UNIMPLEMENT
#endif
}
}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
