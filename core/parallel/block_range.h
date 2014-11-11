/**
 * \file block_range.h
 *
 * \date    2014年9月5日  下午2:57:34 
 * \author salmon
 */

#ifndef BLOCK_RANGE_H_
#define BLOCK_RANGE_H_

#include <type_traits>

#include "../utilities/sp_type_traits.h"
#include "../utilities/ntuple.h"

namespace simpla
{

template<typename T, size_t NDIMS = 1>
class BlockRange
{
public:
	typedef BlockRange<T, NDIMS> this_type;

	static constexpr size_t ndims = NDIMS;

	typedef T single_index_type;

	typedef typename std::conditional<NDIMS == 1, T,
			nTuple<single_index_type, ndims>>::type index_type;

	BlockRange(index_type b = 0, index_type e = 1) :
			e_(e), b_(b)
	{
	}

	//! Copy constructor
	BlockRange(this_type const & that) :
			e_(that.e_), b_(that.b_)
	{
	}
	//! Split range r into two subranges. [0,d) -> [0,n) [n,d)
	BlockRange(this_type & r, //
			size_t n = 1, // numerator
			size_t d = 2 // denominator
			) :
			e_(r.e_), b_(do_split(r.b_, r.e_, n, d))
	{
	}

	//! Split range r into two subranges.

	~BlockRange()  //! Destructor
	{

	}

	void swap(this_type & r)
	{
		std::swap(e_, r.e_);
		std::swap(b_, r.b_);
	}

	bool empty() const //! True if range is empty
	{
		return e_ <= b_;
	}

	bool is_divisible() const //!True if range can be partitioned into two subranges
	{
		return !((e_ - b_) <= 1);
	}

/// \ingroup container_range

private:
	index_type e_, b_;

	static index_type do_split(index_type & b, index_type & e, size_t n = 1,
			size_t d = 2, size_t primary_dim = 0)
	{
		ASSERT((e - b) > 1);

		index_type res = e;

		get_value(res, primary_dim) = get_value(b, primary_dim)
				+ (get_value(e, primary_dim) - get_value(b, primary_dim)) * n
						/ d;

		get_value(b, primary_dim) = get_value(res, primary_dim);

		return std::move(res);
	}

}
;

}
// namespace simpla

#endif /* BLOCK_RANGE_H_ */
