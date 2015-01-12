/*
 * blocked_range.h
 *
 *  Created on: 2014年12月10日
 *      Author: salmon
 */

#ifndef CORE_PARALLEL_BLOCKED_RANGE_H_
#define CORE_PARALLEL_BLOCKED_RANGE_H_
#include "../utilities/type_traits.h"
namespace simpla
{
class split;
template<typename T = size_t, size_t N = 1>
struct BlockedRange;

template<typename T>
struct BlockedRange<T, 1>
{
public:
	typedef BlockedRange<T, 1> this_type;
	typedef size_t size_type;
	typedef T const_iterator;
	typedef T value_type;

	// constructors
	BlockedRange(const const_iterator first, const const_iterator last,
			const size_type grainsize = 1000) :
			begin_(first), end_(last), grainsize_(grainsize)
	{
	}

	BlockedRange(this_type const &other) :
			end_(other.end_), begin_(other.begin_), grainsize_(other.grainsize_)
	{
	}

	BlockedRange(BlockedRange<T> &r, split) :
			end_(r.end_), begin_(r.begin_), grainsize_(r.grainsize_)
	{
		const_iterator beginning = r.begin_, ending = r.end_, middle = beginning
				+ (ending - beginning) / 2u;

		r.end_ = begin_ = middle;
	}

	void swap(this_type & other)
	{
		std::swap(begin_, other.begin_);
		std::swap(end_, other.end_);
		std::swap(grainsize_, other.grainsize_);
	}
	// iterators
	const_iterator begin() const
	{
		return begin_;
	}

	const_iterator end() const
	{
		return end_;
	}

	// access
	bool is_divisible() const
	{
		return ((begin_ + this->grainsize()) < end_);
	}
	size_type grainsize() const
	{
		return grainsize_;
	}

	void grainsize(size_type const &gs)
	{
		grainsize_ = gs;
	}

	// capacity
	int size() const
	{
		return (end_ - begin_);

	}

	bool empty() const
	{
		return (begin_ == end_);
	}

private:

	const_iterator end_;
	const_iterator begin_;
	size_type grainsize_;

};



/** @{*/
struct iterator;

struct range
{

	typedef typename StructuredMesh::iterator iterator;

	index_tuple begin_, end_;
	id_type shift_ = 0UL;

	range()
	{
	}

	range(index_tuple const & b, index_tuple const& e, id_type shift) :
			begin_(b), end_(e), shift_(shift)
	{
	}

	range(range const & that) :
			begin_(that.begin_), end_(that.end_), shift_(that.shift_)
	{
	}
	~range()
	{
	}

	iterator begin() const
	{
		return iterator(begin_, end_, shift_);
	}

	iterator end() const
	{
		iterator e(begin_, end_, shift_);
		e.NextCell();
		return std::move(e);
	}

	iterator rbegin() const
	{
		return iterator(begin_, end_, shift_);
	}

	iterator rend() const
	{

		iterator e(begin_, end_, shift_);
		--e;
		return std::move(e);
	}

}; //struct range

//! iterator
struct iterator
{
	typedef std::bidirectional_iterator_tag iterator_category;

	typedef id_type value_type;

	id_type shift_;

	nTuple<index_type, ndims> self_, begin_, end_;

	iterator(iterator const & r) :
			shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}
	iterator(iterator && r) :
			shift_(r.shift_), self_(r.self_), begin_(r.begin_), end_(r.end_)
	{
	}

	iterator(nTuple<index_type, ndims> const & b,
			nTuple<index_type, ndims> const e, id_type shift = 0UL) :
			shift_(shift), self_(b), begin_(b), end_(e)
	{
	}

	~iterator()
	{
	}

	bool operator==(iterator const & rhs) const
	{
		return self_ == rhs.self_ && shift_ == rhs.shift_;
	}

	bool operator!=(iterator const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	value_type operator*() const
	{
		return compact(self_ << MAX_DEPTH_OF_TREE) | shift_;
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

#ifndef USE_FORTRAN_ORDER_ARRAY
	static constexpr size_t ARRAY_ORDER = C_ORDER;
#else
	static constexpr size_t ARRAY_ORDER=FOTRAN_ORDER;
#endif

	void NextCell()
	{
		CHECK(self_);
		CHECK(begin_);
		CHECK(end_);
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

	void PreviousCell()
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
public:

	void next()
	{
		CHECK(shift_);
		auto n = node_id(shift_);

		if (n == 0 || n == 1 || n == 6 || n == 7)
		{
			NextCell();
		}

		shift_ = roate(shift_);
	}
	void prev()
	{
		auto n = node_id(shift_);

		if (n == 0 || n == 4 || n == 3 || n == 7)
		{
			PreviousCell();
		}

		shift_ = inverse_roate(shift_);
	}
};

}  // namespace simpla

#endif /* CORE_PARALLEL_BLOCKED_RANGE_H_ */
