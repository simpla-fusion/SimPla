/*
 * sp_iterator_index_base.h
 *
 *  created on: 2014-6-20
 *      Author: salmon
 */

#ifndef SP_ITERATOR_INDEX_BASE_H_
#define SP_ITERATOR_INDEX_BASE_H_

#include "Iterator.h"
namespace simpla
{

template<typename T>
class Iterator<T*, std::nullptr_t, std::nullptr_t> : public std::iterator<std::bidirectional_iterator_tag, T>
{
public:
	typedef Iterator<T*, std::nullptr_t, std::nullptr_t> this_type;

	Iterator()
	{
	}
	virtual ~Iterator()
	{
	}
	virtual T get() const =0;
	virtual void next()=0;
	virtual void prev()=0;
	virtual bool is_same(this_type const& rhs) const
	{
		return this->get() == rhs.get();
	}

	auto operator*() const DECL_RET_TYPE (this->get())

	this_type & operator++()
	{
		next();
		return *this;
	}
	this_type & operator--()
	{
		prev();
		return *this;
	}

	bool operator==(this_type const &rhs) const
	{
		return is_same(rhs);
	}
	bool operator!=(this_type const &rhs) const
	{
		return !is_same(rhs);
	}

};
}  // namespace simpla

#endif /* SP_ITERATOR_INDEX_BASE_H_ */
