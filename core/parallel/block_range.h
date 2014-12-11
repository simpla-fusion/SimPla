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

template<typename TI, size_t N>
struct BlockRange
{
	typedef nTuple<TI, N> value_type;

	nTuple<TI, N> begin_, end_;

	struct iterator;

	BlockRange()
	{
	}

	BlockRange(BlockRange const & that) :
			begin_(that.begin_), end_(that.end_)
	{
	}
	~BlockRange()
	{
	}
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

	iterator begin() const
	{
		return iterator(*this, begin_);
	}

	iterator end() const
	{
		iterator e(*this, end_ - 1);
		++e;
		return std::move(e);
	}





};
//! iterator
template<typename TI, typename TOP>
void foreach(BlockRange<TI, 3> const & range, TOP const & op)
{
	for (TI i0 = range.begin_[0]; i0 < range.end_[0]; ++i0)
		for (TI i1 = range.begin_[1]; i1 < range.end_[1]; ++i0)
			for (TI i2 = range.begin_[2]; i2 < range.end_[2]; ++i0)
			{
				nTuple<TI, 3> it = { i0, i1, i2 };
				op(it);
			}
}

template<typename TI, typename TOP>
void foreach(BlockRange<TI, 2> const & range, TOP const & op)
{
	for (TI i0 = range.begin_[0]; i0 < range.end_[0]; ++i0)
		for (TI i1 = range.begin_[1]; i1 < range.end_[1]; ++i0)

		{
			nTuple<TI, 2> it = { i0, i1 };
			op(it);
		}
}

template<typename TI, typename TOP>
void foreach(BlockRange<TI, 1> const & range, TOP const & op)
{
	for (TI i0 = range.begin_[0]; i0 < range.end_[0]; ++i0)
	{
		op(i0);
	}
}

template<typename TI, size_t N>
struct BlockRange<TI, N>::iterator
{
	typedef BlockRange<TI, N> range_type;

	typedef std::bidirectional_iterator_tag iterator_category;

	typedef typename range_type::value_type value_type;

	typedef value_type difference_type;

	static constexpr size_t ndims = N;

	value_type self_, begin_, end_;

	iterator(value_type const & b, value_type const & e) :
			self_(b), begin_(b), end_(e)
	{

	}

	iterator(iterator const & r) :
			self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}
	iterator(iterator && r) :
			self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}

	~iterator()
	{
	}

	bool operator==(iterator const & rhs) const
	{
		return self_ == rhs.self_;
	}

	bool operator!=(iterator const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	value_type const & operator*() const
	{
		return self_;
	}
	value_type & operator*()
	{
		return self_;
	}
	iterator const * operator->() const
	{
		return this;
	}
	iterator * operator->()
	{
		return this;
	}

	iterator & operator ++()
	{
		next();
		return *this;
	}
	iterator operator ++(int) const
	{
		iterator res(*this);
		++res;
		return std::move(res);
	}

	iterator & operator --()
	{
		prev();
		return *this;
	}

	iterator operator --(int) const
	{
		iterator res(*this);
		--res;
		return std::move(res);
	}
private:
	void next() const
	{
#ifndef USE_FORTRAN_ORDER_ARRAY
		++self_[ndims - 1];

		for (int i = ndims - 1; i > 0; --i)
		{
			if (self_[i] >= end_[i])
			{
				self_[i] = begin_[i];
				++self_[i - 1];
			}
		}
#else
		++self_[0];

		for (int i = 0; i < ndims - 1; ++i)
		{
			if (self_[i] >= end_[i])
			{
				self_[i] = begin_[i];
				++self_[i + 1];
			}
		}
#endif
	}

	void previous() const
	{
#ifndef USE_FORTRAN_ORDER_ARRAY

		if (self_[ndims - 1] > begin_[ndims - 1])
			--self_[ndims - 1];

		for (int i = ndims - 1; i > 0; --i)
		{
			if (self_[i] <= begin_[i])
			{
				self_[i] = end_[i] - 1;

				if (self_[i - 1] > begin_[i - 1])
					--self_[i - 1];
			}
		}

#else

		++self_[0];

		for (int i = 0; i < ndims; ++i)
		{
			if (self_[i] < begin_[i])
			{
				self_[i] = end_[i] - 1;
				--self_[i + 1];
			}
		}

#endif //USE_FORTRAN_ORDER_ARRAY
	}
};

}
// namespace simpla

#endif /* BLOCK_RANGE_H_ */
