/**
 * @file iterator_proxy.h
 * @author salmon
 * @date 2015-07-29.
 */

#ifndef SIMPLA_ITERATOR_PROXY_H
#define SIMPLA_ITERATOR_PROXY_H

#include "../type_traits.h"

namespace simpla
{

template<typename TD, typename TIterator>
struct iterator_proxy : public TIterator
{
	typedef TD data_container_type;

	typedef TIterator base_iterator;

private:
	traits::reference_t<data_container_type> m_data_;

public:

	iterator_proxy(this_type &d, base_iterator &it) :
			base_iterator(it), m_view_(d)
	{

	}

	iterator_proxy(const iterator_proxy &other) : base_iterator(other), m_data_(other.m_data_)
	{

	}

	virtual ~iterator()
	{
	}

	using base_iterator::operator==();
	using base_iterator::operator!=();
	using base_iterator::operator++();

	auto  operator*()
	DECL_RET_TYPE((m_data_[base_iterator::operator*()]))


	auto  operator->()
	DECL_RET_TYPE(&(m_data_[base_iterator::operator*()]))

};

template<typename TD, typename TIterator>
iterator_proxy<TD, TIterator> make_iterator_proxy(TD &d, TIterator const &it)
{
	return iterator_proxy<TD, TIterator>(d, it);

};
}//namespace simpla
#endif //SIMPLA_ITERATOR_PROXY_H
