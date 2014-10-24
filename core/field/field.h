/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */
#ifndef CORE_FIELD_FIELD_H_
#define CORE_FIELD_FIELD_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>
#include "field_traits.h"

#include "../utilities/container_traits.h"
#include "../utilities/expression_template.h"
#include "../utilities/sp_type_traits.h"
#include "../parallel/parallel.h"

namespace simpla
{

/**
 *  \brief Field concept
 */
template<typename ... >struct _Field;

/**
 *
 *  \brief skeleton of Field data holder
 *   Field is a Associate container
 *     f[index_type i] => value at the discrete point/edge/face i
 *   Field is a Function
 *     f(coordinates_type x) =>  field value(scalar/vector/tensor) at the coordinates x
 *   Field is a Expression
 */
template<typename TDomain, typename Container>
struct _Field<TDomain, Container>
{

	typedef TDomain domain_type;
	typedef typename domain_type::index_type index_type;
	typedef Container container_type;
	typedef _Field<domain_type, container_type> this_type;
	typedef typename container_traits<container_type>::value_type value_type;

private:

	domain_type domain_;

	container_type data_;

public:

	template<typename ...Args>
	_Field(domain_type const & domain, Args &&...args) :
			domain_(domain), data_(std::forward<Args>(args)...)
	{
	}
	_Field(this_type const & that) :
			domain_(that.domain_), data_(that.data_)
	{
	}

	~_Field()
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
		return data_ == r.data_;
	}
	template<typename TR>
	bool is_same(TR const & r) const
	{
		return data_ == r.data_;
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

	void clear()
	{
		container_traits<container_type>::clear(data_, size());
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

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s]= get_value(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] +=get_value(that, s);
		});

		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{
		allocate();
		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] -= get_value(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] *= get_value(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] /= get_value(that, s);
		});

		return (*this);
	}
///@}

	template<typename ...Args>
	auto operator()(Args && ... args) const
	DECL_RET_TYPE((simpla::gather(domain_,data_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ... args) const
	DECL_RET_TYPE(( simpla::gather(domain_,data_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ... args)
	DECL_RET_TYPE(( simpla::scatter(domain_,data_,
							std::forward<Args>(args)...)))

}
;

/// \defgroup   Field Expression
/// @{
template<typename ... >struct Expression;

/**
 *     \brief skeleton of Field expression
 */

template<typename TOP, typename TL, typename TR>
struct _Field<Expression<TOP, TL, TR>>
{
	typename _impl::reference_traits<TL>::type lhs;
	typename _impl::reference_traits<TR>::type rhs;

	TOP op_;

	_Field(TL const & l, TR const & r) :
			lhs(l), rhs(r), op_()
	{
	}
	_Field(TOP op, TL const & l, TR const & r) :
			lhs(l), rhs(r), op_(op)
	{
	}

	~_Field()
	{
	}
	operator bool() const
	{
//		auto d = get_domain(*this);
//		return   parallel_reduce<bool>(d, _impl::logical_and(), *this);
		return false;
	}
	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((op_( lhs, rhs, s )))

}
;

///   \brief  Unary operation
template<typename TOP, typename TL>
struct _Field<Expression<TOP, TL>>
{

	typename _impl::reference_traits<TL>::type lhs;

	TOP op_;

	_Field(TOP op, TL const & l) :
			lhs(l), op_(op)
	{
	}

	_Field(TL const & l) :
			lhs(l), op_()
	{
	}

	~_Field()
	{
	}

	operator bool() const
	{
		//		auto d = get_domain(*this);
		//		return   parallel_reduce<bool>(d, _impl::logical_and(), *this);
		return false;
	}

	template<typename IndexType>
	inline auto operator[](IndexType const &s) const
	DECL_RET_TYPE ((op_( lhs, s) ))

};

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)
/// @}

template<typename TDomain, typename TV> using Field= _Field<TDomain, std::shared_ptr<TV> >;

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_H_ */
