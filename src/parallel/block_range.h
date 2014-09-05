/**
 * \file block_range.h
 *
 * \date    2014年9月5日  下午2:57:34 
 * \author salmon
 */

#ifndef BLOCK_RANGE_H_
#define BLOCK_RANGE_H_
#include "../utilities/range.h"

namespace simpla
{
/**
 *   is compatible with TBB block_range  1D,2D,3D
 */
template<typename T, unsigned int N = 1> class BlockRange;

template<typename T>
class BlockRange<T, 1>
{

	typedef T const_iterator;
	typedef T value_type;

	//! Type for size of a range
	typedef std::size_t size_type;

	static constexpr unsigned ndims = 1;

	typedef BlockRange<T, 1> this_type;

	/// \ingroup conecpt_range
	BlockRange(this_type const &)  //! Copy constructor
			: i_e_(), i_b_(),

			o_e_(i_e_),

			o_b_(i_b_),

			g_e_(i_e_),

			g_b_(i_b_),

			grainsize_(1),

			ghostwidth_(0)
	{
	}

	BlockRange(this_type & r, split) //! Split range r into two subranges.
			: i_e_(r.i_e_), i_b_(do_split(r, split())),

			o_e_(r.i_e_),

			o_b_(i_b_ - r.ghostwidth_),

			g_e_(r.g_e_),

			g_b_(r.g_e_),

			grainsize_(grainsize),

			ghostwidth_(r.ghostwidth_)

	{
	}

	BlockRange(value_type b, value_type e, size_type grainsize = 1, size_type ghost_width = 2) //! Split range r into two subranges.
			: i_e_(e), i_b_(b),

			o_e_(e + ghost_width),

			o_b_(b - ghost_width),

			g_e_(e),

			g_b_(b),

			grainsize_(grainsize), ghostwidth_(ghost_width)
	{
	}
	~BlockRange()  //! Destructor
	{

	}

	bool empty() const //! True if range is empty
	{
		return i_e_ <= i_b_;
	}

	bool is_divisible() const //!True if range can be partitioned into two subranges
	{
		return grainsize_ > size();
	}
/// \ingroup container_range

	const_iterator begin() const
	{
		return i_b_;
	}

	const_iterator end() const
	{
		return i_e_;
	}
	size_type size() const
	{
		return size_type(end() - begin());
	}

	const_iterator outter_begin() const
	{
		return o_b_;
	}

	const_iterator outter_end() const
	{
		return o_e_;
	}

	size_type memory_size() const
	{
		return size_type(end() - begin());
	}

	const_iterator global_begin() const
	{
		return g_b_;
	}

	const_iterator global_end() const
	{
		return g_e_;
	}
	size_type global_size() const
	{
		return size_type(global_end() - global_begin());
	}

	size_type hash(value_type const &i) const
	{
		return i - i_b_;
	}
private:
	value_type i_e_, i_b_;
	value_type o_e_, o_b_;
	value_type g_e_, g_b_;
	size_type grainsize_;
	size_type ghostwidth_;

	static value_type do_split(this_type & r, split)
	{
		ASSERT(r.is_divisible());

		value_type m = r.i_b_ + (r.i_e_ - r.i_b_) / 2u;
		r.i_e_ = m;
		r.o_e_ = m + r.ghostwidth_;
		return m;
	}

}
;
template<typename T, unsigned int N>
class BlockRange<T, N>
{
public:

	typedef T const_iterator;

	static constexpr unsigned ndims = N;

	typedef BlockRange<T, N> this_type;

	/// \ingroup conecpt_range
	BlockRange(this_type const &); //! Copy constructor

	BlockRange(this_type & r, split); //! Split range r into two subranges.

	~BlockRange(); //! Destructor

	bool empty() const; //! True if range is empty

	bool is_divisible() const //!True if range can be partitioned into two subranges

	/// \ingroup BlockRange
private:
	unsigned int ndims = 3;

};

}  // namespace simpla

#endif /* BLOCK_RANGE_H_ */
