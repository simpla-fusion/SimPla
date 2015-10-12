/**
 * @file tiling_block.h
 * @author salmon
 * @date 2015-07-31.
 */

#ifndef SIMPLA_TILING_BLOCK_H
#define SIMPLA_TILING_BLOCK_H

#include <vector>
#include <memory>
#include "../../gtl/ntuple.h"

namespace simpla
{

template<int NDIMS>
struct TilingBlock
{
	typedef nTuple<size_t, NDIMS> index_tuple;
	typedef TilingBlock<ndims> this_type;
	typedef TilingBlock<ndims> block_type;

	index_tuple m_grain_block_;

	index_tuple m_split_proportion_;

	struct block_s
	{
		index_tuple min;
		index_tuple max;
	};

	std::vector<std::shared_ptr<block_type>> m_children_;


	size_t num_of_children() const;

	std::tuple<index_tuple, index_tuple> box_of_child(int i) const;

	std::tuple<index_tuple, index_tuple> box() const;




};
}// namespace simpla

#endif //SIMPLA_TILING_BLOCK_H
