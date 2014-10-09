/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <cstdbool>

#include "../utilities/expression_template.h"
#include "../utilities/sp_type_traits.h"

namespace simpla
{
/**
 *  \brief traits
 */
template<typename T> struct field_traits;

/**
 *  \brief Field concept
 */

template<typename ...T> struct _Field;

/**
 *
 *  \brief skeleton of Field data holder
 */
template<typename TV, typename TDomain,
		template<typename > class DataContainerPolicy,
		typename ...OthersPolicies>
struct _Field<TV, TDomain, DataContainerPolicy<TV>, OthersPolicies ...> : public OthersPolicies ...
{

	typedef TV value_type;
	typedef TDomain domain_type;
	typedef typename domain_type::index_type index_type;
	typedef typename domain_type::coordinates_type coordinates_type;

	typedef DataContainerPolicy<value_type> storage_policy;
	typedef FieldFunctionPolicy<domain_type> field_function_policy;

	typedef _Field<domain_type, value_type, storage_policy,
			field_function_policy, OthersPolicies ...> this_type;

	domain_type domain_;

	storage_policy data_;

	_Field() = delete;

	_Field(domain_type const & d) :
			domain_(d), data_(nullptr)
	{
	}

	template<typename ...Args>
	_Field(domain_type const & d, Args &&... args) :
			domain_(d), data_(std::forward<Args>(args)...)
	{
	}
	_Field(this_type const & r) :
			domain_(r.domain_), data_(r.data_)
	{
	}
	~_Field()
	{
	}

	void swap(this_type r)
	{
		simpla::swap(r.data_, data_);
		simpla::swap(r.domain_, domain_);
	}

	bool is_same(this_type const & r) const
	{
		return data_ == r.data_;
	}
	template<typename TR>
	bool is_same(TR const &) const
	{
		return false;
	}
	void allocate()
	{
		if (!data_)
			storage_policy(domain_.max_hash()).swap(data_);
	}

	storage_policy& data()
	{
		return data_;
	}

	storage_policy const& data() const
	{
		return data_;
	}

	void data(storage_policy d)
	{
		simpla::swap(d, data_);
	}

	domain_type const & domain() const
	{
		return domain_;
	}

	void domain(domain_type d) const
	{
		d.swap(domain_);
	}
	/// @defgroup Access operation
	/// @{

	value_type & operator[](index_type const & s)
	{
		return get_value(data_, domain_traits<domain_type>::hash(domain_, s));
	}
	value_type & operator[](index_type const & s) const
	{
		return get_value(data_, domain_.hash(s));
	}

///@}
/// @defgroup Assignment
/// @{
	template<typename TR> inline this_type &
	operator =(TR const &rhs)
	{
		parallel_for_each(domain_ & get_domain(rhs), _impl::_assign(), *this,
				rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &rhs)
	{
		parallel_for_each(domain_ & get_domain(rhs), _impl::plus_assign(),
				*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator -=(TR const &rhs)
	{
		parallel_for_each(domain_ & get_domain(rhs), _impl::minus_assign(),
				*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &rhs)
	{
		parallel_for_each(domain_ & get_domain(rhs), _impl::multiplies_assign(),
				*this, rhs);
		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &rhs)
	{
		parallel_for_each(domain_ & get_domain(rhs), _impl::divides_assign(),
				*this, rhs);
		return (*this);
	}
///@}

/// \defgroup Function
/// @{

	template<typename ... Args>
	inline void scatter(Args && ... args)
	{
		domain_.scatter(data_, std::forward<Args>(args)...);
	}

	template<typename ... Args>
	inline auto gather(Args && ... args) const
	DECL_RET_TYPE(( domain_.gather( data_, std::forward<Args>(args)... )))

	template<typename ... Args>
	inline auto operator()(Args && ... args) const
	DECL_RET_TYPE((domain_.gather( data_, std::forward<Args>(args)... )))

/// @}
};

/**
 *     \brief skeleton of Field expression
 *
 */
template<typename ... T>
struct _Field<Expression<T...>> : public Expression<T...>
{

	operator bool() const
	{
		auto d = get_domain(*this);
		return d && parallel_reduce(d, _impl::logical_and(), *this);
	}

	using Expression<T...>::Expression;

};

}
// namespace simpla

#endif /* FIELD_H_ */
