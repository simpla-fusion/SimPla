/*
 * domain_dummy.h
 *
 *  Created on: 2014年10月3日
 *      Author: salmon
 */

#ifndef DOMAIN_DUMMY_H_
#define DOMAIN_DUMMY_H_

namespace simpla
{

class split;

class BlockDomain
{
	typedef BlockDomain this_type;
	typedef size_t index_type;
	typedef size_t iterator;

	iterator b_, e_;

	BlockDomain(iterator b, iterator e) :
			b_(b), e_(e)
	{
		;
	}
	~BlockDomain()
	{
	}
	BlockDomain(BlockDomain & other, split)
	{
		e_ = other.e_;
		other.e_ = (other.b_ + other.e_) / 2;
		b_ = other.e_;
	}
	void swap(this_type & other)
	{
		std::swap(b_, other.b_);
		std::swap(e_, other.e_);
	}
	this_type operator &(this_type const & other) const // \f$D_0 \cap \D_1\f$
	{

		return BlockDomain(std::max(b_, other.b_), std::min(e_, other.e_));
	}
	bool empty() const // True if domain is empty.
	{
		return e_ == b_;
	}

	bool is_divisible() const // True if domain can be partitioned into two sub-domains.
	{
		return e_ > b_ + 1;
	}
	iterator begin() const // First item in domain.
	{
		return b_;
	}

	iterator end() const // One past last item in domain.
	{
		return e_;
	}

	bool operator==(this_type const& other)
	{
		return e_ == other.e_ && b_ == other.b_;
	}

	this_type const & parent() const  // Parent domain
	{
		return *this;
	}
	size_t hash(index_type s) const // get relative  position of grid point  in the memory
	{
		return s - b_;
	}
	size_t max_hash() const	 // get max number of grid points in memory
	{
		return e_ - b_;	 //manifold_.topology_type::max_hash();
	}

};

}  // namespace simpla

#endif /* DOMAIN_DUMMY_H_ */
