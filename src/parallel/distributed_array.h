/*
 * distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_

#include <vector>
#include <functional>
#include <tuple>
#include "../utilities/ntuple.h"
#include "../utilities/singleton_holder.h"
#include "../numeric/geometric_algorithm.h"

namespace simpla
{
struct DistributedArray
{
public:
	unsigned int ndims = MAX_NDIMS_OF_ARRAY;

	int self_id_ = 0;

	struct sub_array_s
	{
		nTuple<MAX_NDIMS_OF_ARRAY, long> outer_begin;
		nTuple<MAX_NDIMS_OF_ARRAY, long> outer_end;
		nTuple<MAX_NDIMS_OF_ARRAY, long> inner_begin;
		nTuple<MAX_NDIMS_OF_ARRAY, long> inner_end;
	};
	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		unsigned int ndims;

		nTuple<MAX_NDIMS_OF_ARRAY, long> send_begin;
		nTuple<MAX_NDIMS_OF_ARRAY, long> send_end;
		nTuple<MAX_NDIMS_OF_ARRAY, long> recv_begin;
		nTuple<MAX_NDIMS_OF_ARRAY, long> recv_end;
	};

	nTuple<MAX_NDIMS_OF_ARRAY, long> global_begin_;
	nTuple<MAX_NDIMS_OF_ARRAY, long> global_end_;
	nTuple<MAX_NDIMS_OF_ARRAY, long> global_strides_;

	sub_array_s local_;

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	DistributedArray()
			: self_id_(0)
	{
	}

	template<int NDIMS, typename TS>
	DistributedArray(nTuple<NDIMS, long> global_begin, nTuple<NDIMS, long> global_end, int num_process = 1,
	        unsigned int process_num = 0, size_t gw = 0, bool p_is_fast_first = false)
			: ndims(NDIMS)
	{

		global_end_ = global_end;
		global_begin_ = global_begin;

		Decompose(gw);
	}

	~DistributedArray()
	{
	}
	size_t size() const
	{
		return NProduct(local_.inner_end - local_.inner_begin);
	}
	size_t memory_size() const
	{
		return NProduct(local_.outer_end - local_.outer_begin);
	}

	void Decompose(size_t gw = 2);

	void Decomposer_(int num_process, unsigned int process_num, unsigned int gw, sub_array_s *) const;

	template<typename TS>
	int hash(TS const * d) const
	{
		int res = 0;
		for (int i = 0; i < ndims; ++i)
		{
			res += ((d[i] - global_begin_[i] + (global_end_[i] - global_begin_[i]))
			        % (global_end_[i] - global_begin_[i])) * global_strides_[i];
		}
		return res;
	}
}
;

}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
