/*
 * distributed_array.h
 *
 *  created on: 2014-5-30
 *      Author: salmon
 */

#ifndef DISTRIBUTED_ARRAY_H_
#define DISTRIBUTED_ARRAY_H_

#include <stddef.h>
#include <vector>

#include "../utilities/data_type.h"
#include "../utilities/ntuple.h"

namespace simpla
{
struct DistributedArray
{
public:
	unsigned int ndims = 3;

	int self_id_ = 0;

	DistributedArray()
			: self_id_(0)
	{
	}

	template<typename TI>
	DistributedArray(TI b, TI e, long gw = 0)
	{
		init(b, e, gw);
	}

	template<typename TI>
	void init(unsigned int nd, TI const & b, TI const & e, unsigned int gw = 2)
	{
		ndims = nd;

		for (int i = 0; i < nd; ++i)
		{
			global_begin_[i] = b[i];
			global_end_[i] = e[i];
		}
		Decompose(gw);
	}
	~DistributedArray()
	{
	}
	size_t size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims; ++i)
		{
			res *= (local_.inner_end[i] - local_.inner_begin[i]);
		}
		return res;
	}
	size_t memory_size() const
	{
		size_t res = 1;

		for (int i = 0; i < ndims; ++i)
		{
			res *= (local_.outer_end[i] - local_.outer_begin[i]);
		}
		return res;
	}

	void Decompose(long gw = 2);

	nTuple<3, long> global_begin_;
	nTuple<3, long> global_end_;
	nTuple<3, long> global_strides_;

	struct sub_array_s
	{
		nTuple<3, long> outer_begin;
		nTuple<3, long> outer_end;
		nTuple<3, long> inner_begin;
		nTuple<3, long> inner_end;
	};
	sub_array_s local_;

	struct send_recv_s
	{
		int dest;
		int send_tag;
		int recv_tag;
		nTuple<3, long> send_begin;
		nTuple<3, long> send_end;
		nTuple<3, long> recv_begin;
		nTuple<3, long> recv_end;
	};

	std::vector<send_recv_s> send_recv_; // dest, send_tag,recv_tag, sub_array_s

	template<typename TI>
	int hash(TI const & d) const
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

void update_ghosts(void * data, DataType const & data_type, DistributedArray const & global_array);

template<typename TV>
void update_ghosts(TV * data, DistributedArray const & global_array)
{
	update_ghosts(data, DataType::create<TV>(), global_array);
}
}
// namespace simpla

#endif /* DISTRIBUTED_ARRAY_H_ */
