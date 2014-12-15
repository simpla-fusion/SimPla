/*
 * range.h
 *
 *  Created on: 2014年12月9日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_RANGE_H_
#define CORE_PARALLEL_RANGE_H_
#include "../utilities/sp_type_traits.h"

namespace simpla
{
template<size_t ND, typename TI = size_t, typename HASH>
struct Range
{

	typedef typename StructuredMesh::iterator iterator;
#ifndef USE_FORTRAN_ORDER_ARRAY
	static constexpr size_t ARRAY_ORDER = C_ORDER;
#else
	static constexpr size_t ARRAY_ORDER=FOTRAN_ORDER;
#endif

	typedef TI index_type;

	nTuple<size_t, ND> begin_, end_;

	Range()
	{
	}

	Range(Range const & that) :
			begin_(that.begin_), end_(that.end_)
	{
	}
	~Range()
	{
	}

	HASH hash;

	auto min_hash() const
	DECL_RET_TYPE(( hash (begin_)))

	auto max_hash() const
	DECL_RET_TYPE(( hash (end_) ))

	auto size() const
	DECL_RET_TYPE((max_hash( ) - min_hash( )))

	template<typename TOP> void
	foreach(TOP const & op);

//private:
//	void NextCell(iterator & it) const
//	{
//#ifndef USE_FORTRAN_ORDER_ARRAY
//		++it.self_[ndims - 1];
//
//		for (int i = ndims - 1; i > 0; --i)
//		{
//			if (it.self_[i] >= end_[i])
//			{
//				it.self_[i] = begin_[i];
//				++it.self_[i - 1];
//			}
//		}
//#else
//		++it.self_[0];
//
//		for (int i = 0; i < ndims - 1; ++i)
//		{
//			if (it.self_[i] >= end_[i])
//			{
//				it.self_[i] = begin_[i];
//				++it.self_[i + 1];
//			}
//		}
//#endif
//	}
//
//	void PreviousCell(iterator & it) const
//	{
//#ifndef USE_FORTRAN_ORDER_ARRAY
//
//		if (it.self_[ndims - 1] > begin_[ndims - 1])
//			--it.self_[ndims - 1];
//
//		for (int i = ndims - 1; i > 0; --i)
//		{
//			if (it.self_[i] <= begin_[i])
//			{
//				it.self_[i] = end_[i] - 1;
//
//				if (it.self_[i - 1] > begin_[i - 1])
//					--it.self_[i - 1];
//			}
//		}
//
//#else
//
//		++it.self_[0];
//
//		for (int i = 0; i < ndims; ++i)
//		{
//			if (it.self_[i] < begin_[i])
//			{
//				it.self_[i] = end_[i] - 1;
//				--it.self_[i + 1];
//			}
//		}
//
//#endif //USE_FORTRAN_ORDER_ARRAY
//	}
//public:
//
//	void next(iterator & it) const
//	{
//		auto n = node_id(it.shift_);
//
//		if (n == 0 || n == 1 || n == 6 || n == 7)
//		{
//			NextCell(it);
//		}
//
//		it.shift_ = roate(it.shift_);
//	}
//	void prev(iterator & it) const
//	{
//		auto n = node_id(it.shift_);
//
//		if (n == 0 || n == 4 || n == 3 || n == 7)
//		{
//			PreviousCell(it);
//		}
//
//		it.shift_ = inverse_roate(it.shift_);
//	}
//
//	iterator begin() const
//	{
//		return iterator(*this, begin_);
//	}
//
//	iterator end() const
//	{
//		iterator e(*this, end_ - 1);
//		NextCell(e);
//		return std::move(e);
//	}
//
//	iterator rbegin() const
//	{
//		return iterator(*this, end_ - 1);
//	}
//
//	iterator rend() const
//	{
//
//		iterator e(*this, begin_ - 1);
//		--e;
//		return std::move(e);
//	}

};

template<typename TOP>
void Range<3, size_t>::foreach(TOP const & op)
{
	for (size_t ib = begin_[0], ie = end_[0]; ib != ie; ++ib)
		for (size_t jb = begin_[1], je = end_[1]; jb != je; ++jb)
			for (size_t kb = begin_[2], ke = end_[2]; kb != ke; ++ib)
			{
				op(mesh->compact(ib, jb, kb));
			}
}

template<typename TOP, typename Reduction> typename std::result_of<TOP(size_t)>::type Range<
		3, size_t>::reduce(TOP const & op, Reduction const & r)
{
	typedef typename std::result_of<TOP(size_t)>::type value_type;

	value_type res;

	return res;

}
//struct range
}// namespace simpla

#endif /* CORE_PARALLEL_RANGE_H_ */
