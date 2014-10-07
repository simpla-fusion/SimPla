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
template<typename TDomain, typename DataContainer>
struct _Field<TDomain, DataContainer>
{

	typedef DataContainer data_container_type;

	typedef TDomain domain_type;

	typedef _Field<domain_type, data_container_type> this_type;

	typedef typename data_container_type::value_type value_type;
	typedef typename domain_type::index_type index_type;
	typedef typename domain_type::coordinates_type coordinates_type;

	domain_type domain_;

	data_container_type data_;

	_Field() = delete;

	_Field(TDomain const & d) :
			domain_(d), data_(nullptr)
	{
	}

	template<typename ...Args>
	_Field(TDomain const & d, Args &&... args) :
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
			data_container_type(domain_.max_hash()).swap(data_);
	}

	data_container_type& data()
	{
		return data_;
	}

	data_container_type const& data() const
	{
		return data_;
	}

	void data(data_container_type d)
	{
		d.swap(data_);
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
		return get_value(data_, domain_.hash(s));
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
	DECL_RET_TYPE( ( domain_.gather( data_, std::forward<Args>(args)... )))

	template<typename ... Args>
	inline auto operator()(Args && ... args) const
	DECL_RET_TYPE( (domain_.gather( data_, std::forward<Args>(args)... )))

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
