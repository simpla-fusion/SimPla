/**
 * @file sp_iterator.h
 *
 * @date 2015-2-12
 * @author salmon
 */

#ifndef CORE_GTL_ITERATOR_SP_ITERATOR_H_
#define CORE_GTL_ITERATOR_SP_ITERATOR_H_

#include <iterator>
#include "../type_traits.h"
namespace simpla
{
namespace gtl {

template<typename > struct sp_back_insert_iterator;

template<typename T>
class sp_back_insert_iterator<T*> : public std::iterator<
		std::output_iterator_tag, T, void, void, void>
{

public:

	typedef T value_type;
	typedef T * container_type;

protected:
	container_type m_p_;
public:
	/// The only way to create this %iterator is with a container.
	explicit sp_back_insert_iterator(container_type p)
			: m_p_(p)
	{
	}
	value_type * get() const
	{
		return m_p_;
	}

	sp_back_insert_iterator&
	operator=(const value_type& __value)
	{
		*m_p_ = __value;
		++m_p_;
		return *this;
	}

	sp_back_insert_iterator&
	operator=(const value_type&& __value)
	{
		*m_p_ = __value;
		++m_p_;
		return *this;
	}

	/// Simply returns *this.
	sp_back_insert_iterator&
	operator*()
	{
		return *this;
	}

	/// Simply returns *this.  (This %iterator does not @a move.)
	sp_back_insert_iterator&
	operator++()
	{
		return *this;
	}

	/// Simply returns *this.  (This %iterator does not @a move.)
	sp_back_insert_iterator operator++(int)
	{
		return *this;
	}
};
template<typename T>
auto back_inserter(T * p)
DECL_RET_TYPE((sp_back_insert_iterator<T*>(p)))
template<typename T>
auto back_inserter(std::shared_ptr<T> p)
DECL_RET_TYPE((sp_back_insert_iterator<T*>(p.get())))

template<typename ...Args>
auto back_inserter(Args && ...args)
DECL_RET_TYPE((std::back_inserter(std::forward<Args>(args)...)))

}  }//  namespace simpla::gtl

#endif /* CORE_GTL_ITERATOR_SP_ITERATOR_H_ */
