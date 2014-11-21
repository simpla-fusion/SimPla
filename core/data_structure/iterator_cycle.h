/*
 * iterator_cycle.h
 *
 *  Created on: 2014年11月19日
 *      Author: salmon
 */

#ifndef CORE_DATA_STRUCTURE_ITERATOR_CYCLE_H_
#define CORE_DATA_STRUCTURE_ITERATOR_CYCLE_H_
#include "../utilities/sp_type_traits.h"
namespace simpla
{
template<typename TI>
struct iterator_cycle
{
	typedef TI base_iterator;
	typedef iterator_cycle<base_iterator> iterator;

	base_iterator p_;
	size_t shift_ = 0, length_ = 1;

	iterator_cycle(base_iterator p, size_t tail = 0, size_t l = 1) :
			p_(p), shift_(tail), length_(l)
	{
	}

	~iterator_cycle()
	{
	}

	size_t cycle_length() const
	{
		return length_;
	}

	bool operator==(iterator const & rhs) const
	{
		return shift_ == rhs.shift_ && p_ == rhs.p_;
	}

	bool operator!=(iterator const & rhs) const
	{
		return !(this->operator==(rhs));
	}

	auto operator*() const
	DECL_RET_TYPE((*(p_+shift_)))

	auto operator->() const
	DECL_RET_TYPE((p_+shift_))

	iterator * operator->()
	DECL_RET_TYPE((p_+shift_))

	iterator & operator ++()
	{
		shift_ = (shift_ + 1) % length_;
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
		shift_ = (shift_ - 1 + length_) % length_;
		return *this;
	}

	iterator operator --(int) const
	{
		iterator res(*this);
		--res;
		return std::move(res);
	}

};
}  // namespace simpla

#endif /* CORE_DATA_STRUCTURE_ITERATOR_CYCLE_H_ */
