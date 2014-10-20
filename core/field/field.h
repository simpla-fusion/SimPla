/*
 * @file field.h
 *
 * @date  2013-7-19
 * @author  salmon
 */

#ifndef FIELD_H_
#define FIELD_H_

#include <stddef.h>
#include <cstdbool>
#include <memory>

#include "../utilities/container.h"
#include "../utilities/expression_template.h"
#include "../utilities/sp_type_traits.h"
#include "../manifold/domain_traits.h"
#include "../parallel/parallel.h"

namespace simpla
{

/**
 *  \brief Field concept
 */
template<typename ... >struct Expression;
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
		for (auto const& s : domain_ & get_domain(that))
		{
			get_value(data_, domain_.hash(s)) = domain_.calcuate(that, s);
		}
//		parallel_for(domain_ & get_domain(that), [&](index_type const & s)
//		{
//			get_value(data_,domain_.hash(s))=domain_.calcuate(that,s);
//		});

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

	template<typename ...Args>
	auto operator()(Args && ... args) const
	DECL_RET_TYPE((gather(domain_,data_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto gather(Args && ... args) const
	DECL_RET_TYPE((gather(domain_, data_,std::forward<Args>(args)...)))

	template<typename ...Args>
	auto scatter(Args && ... args)
	DECL_RET_TYPE((scatter(domain_,data_,std::forward<Args>(args)...)))

}
;

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
/// \defgroup   BasicAlgebra Basic algebra
/// @{
DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)
/// @}

template<typename TDomain, typename TV> using Field=_Field<TDomain, std::shared_ptr<TV> >;
}
// namespace simpla

#endif /* FIELD_H_ */
