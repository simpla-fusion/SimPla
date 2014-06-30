/*
 * iterator_index_base.h
 *
 *  Created on: 2014年6月20日
 *      Author: salmon
 */

#ifndef ITERATOR_INDEX_BASE_H_
#define ITERATOR_INDEX_BASE_H_

#include <iterator>
namespace simpla
{

template<typename T>
class IndexBaseIterator: public std::iterator<std::bidirectional_iterator_tag, T>
{
public:
	typedef IndexBaseIterator<T> this_type;

	IndexBaseIterator()
	{
	}
	virtual ~IndexBaseIterator()
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

#endif /* ITERATOR_INDEX_BASE_H_ */
