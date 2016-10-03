/**
 * @file indirect_iterator.h
 *
 * @date 2015-2-10
 * @author salmon
 */

#ifndef CORE_toolbox_ITERATOR_INDIRECT_ITERATOR_H_
#define CORE_toolbox_ITERATOR_INDIRECT_ITERATOR_H_

namespace simpla
{

template<typename TSIterator>
struct indirect_iterator
{

	typedef TSIterator src_iterator;

	typedef typename std::iterator_traits<src_iterator>::value_type base_iterator;
	typedef typename std::iterator_traits<base_iterator>::value_type value_type;

	typedef indirect_iterator<src_iterator> this_type;

	src_iterator m_it_;

	indirect_iterator(src_iterator const & s_it) :
			m_it_(s_it)
	{

	}

	~indirect_iterator()
	{
	}

	value_type & operator *()
	{
		return **m_it_;
	}

	value_type & operator ->()
	{
		return **m_it_;
	}

	indirect_iterator & operator++()
	{
		++m_it_;
		return *this;
	}
	indirect_iterator operator++(int) const
	{
		indirect_iterator res(*this);
		++res;
		return std::move(res);
	}

	bool operator==(this_type const & other) const
	{
		return m_it_ == other.m_it_;
	}
};

template<typename TSIterator>
constexpr indirect_iterator<TSIterator> make_indirect_iterator(
		TSIterator const & sit)
{
	return std::move(indirect_iterator<TSIterator>(si));
}

}  // namespace simpla

#endif /* CORE_toolbox_ITERATOR_INDIRECT_ITERATOR_H_ */
