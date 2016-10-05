/**
 * @file dataview.h
 * @author salmon
 * @date 2015-07-29.
 */

#ifndef SIMPLA_DATAVIEW_H
#define SIMPLA_DATAVIEW_H

#include <memory>
#include "MemoryPool.h"
#include "../toolbox/containers/iterator_proxy.h"


namespace simpla
{
template<typename ...> struct data_view;

namespace tags
{
struct split;
}

template<typename TV, typename IndexRange, typename ...Tags>
struct data_view<TV, IndexRange, Tags...>
{
	typedef TV value_type;

	typedef IndexRange index_range_type;

	typedef data_view<value_type, index_range_type, Tags...> this_type;

//	typedef iterator_proxy<this_type, traits::iterator_t<index_range_type>> iterator;
//
//	typedef iterator_proxy<const this_type, traits::iterator_t<index_range_type> > const_iterator;


private:

	std::shared_ptr<TV> m_data_;

	index_range_type m_index_range_;

public:

	data_view(index_range_type const &r) :
			m_data_(sp_make_shared_array<value_type>(r.max_hash())), m_index_range_(r)
	{

	}

	data_view(this_type &other, tags::split) :
			m_data_(other.m_data_), m_index_range_(other.m_data_, tags::split())
	{

	}

	data_view(this_type const &other) :
			m_data_(other.m_data_), m_index_range_(other.m_index_range_)
	{

	}

	this_type operator=(this_type const &other)
	{
		this_type(other).swap(*this);

		return *this;
	}

	void swap(this_type &other)
	{

		std::swap(m_data_, other.m_data_);
		std::swap(m_index_range_, other.m_index_range_);
	}

	index_range_type const &index_range() const { return m_index_range_; }

	size_t size() const { return m_index_range_.size(); }


	auto begin() DECL_RET_TYPE((make_iterator_proxy(*this, m_index_range_.begin())))

	auto end() DECL_RET_TYPE((make_iterator_proxy(*this, m_index_range_.end())))

	auto cbegin() const DECL_RET_TYPE((make_iterator_proxy(*this, m_index_range_.begin())))

	auto cend() const DECL_RET_TYPE((make_iterator_proxy(*this, m_index_range_.end())))

	template<typename ...Args>
	this_type slice(Args &&...args)
	{
		return this_type(m_data_, m_index_range_.slice(std::forward<Args>(args)...));
	}


	value_type &operator[](size_t s) { return m_data_.get()[s]; }

	value_type const &operator[](size_t s) const { return m_data_.get()[s]; }


	template<typename ...Args>
	value_type &operator()(Args &&... args)
	{
		return operator[](m_index_range_.hash(std::forward<Args>(args)...));
	}

	template<typename ...Args>
	value_type &operator()(Args &&... args) const
	{
		return operator[](m_index_range_.hash(std::forward<Args>(args)...));
	}
};
}
#endif //SIMPLA_DATAVIEW_H
