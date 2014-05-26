/*
 * decompose.h
 *
 *  Created on: 2014年5月20日
 *      Author: salmon
 */

#ifndef DECOMPOSE_H_
#define DECOMPOSE_H_

#include <stddef.h>
#include <utility>

#include "../fetl/ntuple.h"

namespace simpla
{
template<int NDIMS>
void Decompose(size_t num_process, size_t process_num, size_t gw,

size_t *global_start, size_t *global_count, size_t *local_outer_start, size_t *local_outer_count

, size_t *local_inner_start, size_t *local_inner_count)
{

	for (int i = 0; i < NDIMS; ++i)
	{

		if (num_process[i] <= 1)
		{
			global_start[i] = 0;
			global_end[i] = dims[i];
			local_start[i] = 0;
			local_count[i] = dims[i];

		}
		else if (2 * gw[i] * num_process[i] > dims[i])
		{
			ERROR << "Mesh is too small to decompose! dims[" << i << "]=" << dims[i]

			<< " process[" << i << "]=" << num_process[i] << " ghost_width=" << gw[i];
		}
		else
		{
			global_start[i] = dims[i] * process_num[i] / (num_process[i]) - gw[i];
			global_end[i] = dims[i] * (process_num[i] + 1) / (num_process[i]) + gw[i];

			local_start[i] = global_start[i] + gw[i];
			local_count[i] = global_end[i] - global_start[i] - 2 * gw[i];
		}

	}

}
std::pair<Range, Range> Split2(unsigned int total, unsigned int sub, unsigned int gw = 0) const
{
	std::pair<Range, Range> res;
	nTuple<NDIMS, size_type> num_process;
	nTuple<NDIMS, size_type> process_num;
	nTuple<NDIMS, size_type> ghost_width;

	auto extents = Extents();

	bool flag = false;
	for (int i = 0; i < NDIMS; ++i)
	{
		ghost_width[i] = gw;
		if (!flag && (extents[i] > total))
		{
			num_process[i] = total;
			process_num[i] = sub;
			flag = true;
		}
		else
		{
			num_process[i] = 1;
			process_num[i] = 0;
		}
	}
	if (!flag)
	{
		if (sub == 0)
		{
			WARNING << "I'm the master!";
			res = std::pair<Range, Range>(*this, *this);
		}
		else
		{
			WARNING << "Range is too small to split!  ";
		}
	}
	else
	{
		res = Split2(num_process, process_num, ghost_width);
	}

	return res;

}

std::pair<Range, Range> Split2(nTuple<NDIMS, size_type> const & num_process,
		nTuple<NDIMS, size_type> const & process_num, nTuple<NDIMS, size_type> const & ghost_width) const
{

	nTuple<NDIMS, size_type>

	inner_start = start_,

	inner_count = count_,

	outer_start, outer_count;

	for (int i = 0; i < NDIMS; ++i)
	{

		if (2 * ghost_width[i] * num_process[i] > inner_count[i])
		{
			ERROR << "Mesh is too small to decompose! dims[" << i << "]=" << inner_count[i]

			<< " process[" << i << "]=" << num_process[i] << " ghost_width=" << ghost_width[i];
		}
		else
		{

			auto start = (inner_count[i] * process_num[i]) / num_process[i];

			auto end = (inner_count[i] * (process_num[i] + 1)) / num_process[i];

			inner_start[i] += start;
			inner_count[i] = end - start;

			outer_start[i] = inner_start[i];
			outer_count[i] = inner_count[i];

			if (process_num[i] > 0)
			{
				outer_start[i] -= ghost_width[i];
				outer_count[i] += ghost_width[i];

			}
			if (process_num[i] < num_process[i] - 1)
			{
				outer_count[i] += ghost_width[i];

			};

		}
	}

	return std::make_pair(Range(outer_start, outer_count, shift_), Range(inner_start, inner_count, shift_));
}

}  // namespace simpla

#endif /* DECOMPOSE_H_ */
