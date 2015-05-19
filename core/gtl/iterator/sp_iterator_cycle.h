/**
 * @file sp_iterator_cycle.h
 *
 * @date 2015年5月12日
 * @author salmon
 */

#ifndef CORE_GTL_ITERATOR_SP_ITERATOR_CYCLE_H_
#define CORE_GTL_ITERATOR_SP_ITERATOR_CYCLE_H_
#include <iterator>
namespace simpla
{

template<typename TBaseIterator>
struct CycleIterator: public TBaseIterator
{
	typedef TBaseIterator base_iterator;
	typedef CycleIterator<base_iterator> this_type;
private:
	base_iterator m_begin_, m_end_;
public:

	CycleIterator(base_iterator const & b, base_iterator const &e)
			: base_iterator(b), m_begin_(b), m_end_(e)
	{
	}

	CycleIterator(base_iterator const &self, base_iterator const & b,
			base_iterator const &e)
			: base_iterator(self), m_begin_(b), m_end_(e)
	{
	}
	CycleIterator(this_type const & other)
			: base_iterator(other), m_begin_(other.m_begin_), m_end_(
					other.m_end_)
	{
	}
	CycleIterator(this_type && other)
			: base_iterator(other), m_begin_(other.m_begin_), m_end_(
					other.m_end_)
	{
	}
	~CycleIterator()
	{
	}
	void swap(this_type & other)
	{
		std::swap(*dynamic_cast<base_iterator*>(*this),
				dynamic_cast<base_iterator&>(other));
		std::swap(m_begin_, other.m_begin_);
		std::swap(m_end_, other.m_end_);
	}

	this_type & operator=(this_type & other)
	{
		this_type(other).swap(*this);
		return *this;
	}

	using base_iterator::operator*;
	using base_iterator::operator->;
	using base_iterator::operator==;
	using base_iterator::operator!=;

	this_type & operator++()
	{
		base_iterator::operator++();

		if (base_iterator::operator==(m_end_))
		{
			base_iterator::operator=(m_begin_);
		}

		return *this;
	}

	this_type operator++(int) const
	{
		this_type res(*this);
		++res;
		return std::move(res);

	}
	this_type & operator--()
	{

		if (base_iterator::operator==(m_begin_))
		{
			base_iterator::operator=(m_end_);
		}

		base_iterator::operator--();

		return *this;
	}
	this_type operator--(int) const
	{
		this_type res(*this);
		--res;
		return std::move(res);

	}
};

template<typename TIterator>
CycleIterator<TIterator> make_cycle_iterator(TIterator const & b,
		TIterator const &e)
{
	return CycleIterator<TIterator>(b, e);
}
template<typename TIterator>
CycleIterator<TIterator> make_cycle_iterator(TIterator const & self,
		TIterator const & b, TIterator const &e)
{
	return CycleIterator<TIterator>(self, b, e);
}
}  // namespace simpla

#endif /* CORE_GTL_ITERATOR_SP_ITERATOR_CYCLE_H_ */
