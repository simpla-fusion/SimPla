/*
 * geometric_algorithm.h
 *
 *  Created on: 2014年6月10日
 *      Author: salmon
 */

#ifndef GEOMETRIC_ALGORITHM_H_
#define GEOMETRIC_ALGORITHM_H_

#include <stddef.h>

#include "../fetl/ntuple.h"

namespace simpla
{
template<int NDIMS>
bool Clipping(nTuple<NDIMS, size_t> const & l_start, nTuple<NDIMS, size_t> const &l_count,
        nTuple<NDIMS, size_t> *pr_start, nTuple<NDIMS, size_t> *pr_count)
{
	bool has_overlap = false;

	nTuple<NDIMS, size_t> & r_start = *pr_start;
	nTuple<NDIMS, size_t> & r_count = *pr_count;

	for (int i = 0; i < NDIMS; ++i)
	{
		if (r_start[i] + r_count[i] <= l_start[i] || r_start[i] >= l_start[i] + l_count[i])
			return false;

		size_t start = std::max(l_start[i], r_start[i]);
		size_t end = std::min(l_start[i] + l_count[i], r_start[i] + r_count[i]);

		if (end > start)
		{
			r_start[i] = start;
			r_count[i] = end - start;

			has_overlap = true;
		}
	}

	return has_overlap;
}
}  // namespace simpla

#endif /* GEOMETRIC_ALGORITHM_H_ */
