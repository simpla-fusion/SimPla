/*
 * base_field.h
 *
 *  Created on: Oct 16, 2014
 *      Author: salmon
 */

#ifndef BASE_FIELD_H_
#define BASE_FIELD_H_

#include "../utilities/container.h"
#include "../utilities/sp_type_traits.h"
#include "../manifold/domain_traits.h"
#include "../utilities/expression_template.h"
#include "../parallel/parallel.h"

namespace simpla
{

template<typename TDomain, typename Container>
struct BaseField
{
	typedef TDomain domain_type;

	typedef Container container_type;

	typedef typename container_traits<container_type>::value_type value_type;

	typedef BaseField<domain_type, container_type> this_type;

	domain_type domain_;

	container_type data_;

	BaseField()
	{
	}

	BaseField(domain_type const & d) :
			domain_(d)
	{
	}

	~BaseField()
	{
	}

	void swap(this_type &r)
	{
		sp_swap(r.data_, data_);
		sp_swap(r.domain_, domain_);
	}

	domain_type const & domain() const
	{
		return domain_;
	}

	void domain(domain_type d)
	{
		sp_swap(d, domain_);
	}
	size_t size() const
	{
		return domain_.max_hash();
	}
	container_type & data()
	{
		return data_;
	}

	container_type const & data() const
	{
		return data_;
	}

	void data(container_type d)
	{
		sp_swap(d, data_);
	}

	bool is_same(this_type const & r) const
	{
		return container_traits<container_type>::is_same(data_, r.data_);
	}
	template<typename TR>
	bool is_same(TR const & r) const
	{
		return container_traits<container_type>::is_same(data_, r);
	}

	bool empty() const
	{
		return container_traits<container_type>::is_empty(data_);
	}
	void allocate()
	{
		if (empty())
		{
			container_traits<container_type>::allocate(size()).swap(data_);
		}

	}

	// @defgroup Access operation
	// @

	template<typename TI>
	value_type & operator[](TI const & s)
	{
		return (get_value(data_, domain_.hash(s)));
	}

	template<typename TI>
	value_type const & operator[](TI const & s) const
	{
		return (get_value(data_, domain_.hash(s)));
	}

	///@}

	/// @defgroup Assignment
	/// @{
	template<typename TR> inline this_type &
	operator =(TR const &that)
	{
		allocate();
//
//
//		for (auto const& s : domain_)
//		{
//			this->operator[](s) = get_value(that, s);
//		}

		parallel_for_each(domain_ & get_domain(that), _impl::_assign(), *this,
				that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{
		allocate();
		parallel_for_each(domain_ & get_domain(that), _impl::plus_assign(),
				*this, that);
		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{
		allocate();
		parallel_for_each(domain_ & get_domain(that), _impl::minus_assign(),
				*this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{
		allocate();
		parallel_for_each(domain_ & get_domain(that),
				_impl::multiplies_assign(), *this, that);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{
		allocate();
		parallel_for_each(domain_ & get_domain(that), _impl::divides_assign(),
				*this, that);
		return (*this);
	}
	///@}

};

}
// namespace simpla

#endif /* BASE_FIELD_H_ */
