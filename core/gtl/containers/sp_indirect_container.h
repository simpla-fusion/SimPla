/**
 * @file sp_indirect_container.h
 *
 * @date 2015年2月10日
 * @author salmon
 */

#ifndef CORE_GTL_CONTAINERS_SP_INDIRECT_CONTAINER_H_
#define CORE_GTL_CONTAINERS_SP_INDIRECT_CONTAINER_H_
#include "../iterator/indirect_iterator.h"
namespace simpla
{
template<typename KeyContainer, typename ValueContiner>
struct indirect_container: public KeyContainer
{
	typedef KeyContainer key_conatinaer_type;
	typedef ValueContiner value_conatinaer_type;

	typedef typename value_conatinaer_type::value_type value_type;
	typedef typename key_conatinaer_type::value_type key_type;

	typedef typename key_conatinaer_type::const_iterator const_key_iterator;

	typedef indirect_iterator<const_key_iterator, value_conatinaer_type> iterator;
	typedef indirect_iterator<const_key_iterator, value_conatinaer_type const> const_iterator;

	ValueContiner & m_base_;

	indirect_container(value_conatinaer_type & d) :
			m_base_(d)
	{
	}

	using key_conatinaer_type::insert;
	using key_conatinaer_type::erase;

	iterator begin()
	{
		return iterator(key_conatinaer_type::cbeign(), m_base_);
	}
	iterator ebd()
	{
		return iterator(key_conatinaer_type::cend(), m_base_);
	}
	const_iterator begin() const
	{
		return iterator(key_conatinaer_type::cbeign(), m_base_);
	}
	const_iterator end() const
	{
		return iterator(key_conatinaer_type::cend(), m_base_);
	}
	const_iterator cbegin() const
	{
		return iterator(key_conatinaer_type::cbeign(), m_base_);
	}
	const_iterator cend() const
	{
		return iterator(key_conatinaer_type::cend(), m_base_);
	}
};
}  // namespace simpla

#endif /* CORE_GTL_CONTAINERS_SP_INDIRECT_CONTAINER_H_ */
