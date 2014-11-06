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
#include <tuple>

#include "../utilities/container_traits.h"
#include "../utilities/sp_type_traits.h"
#include "../utilities/data_type.h"
#include "../parallel/parallel.h"
#include "../utilities/expression_template.h"

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
template<typename Container, typename TDomain>
struct _Field<Container, TDomain>
{

	typedef TDomain domain_type;
	typedef typename domain_type::index_type index_type;
	typedef Container container_type;
	typedef _Field<container_type, domain_type> this_type;
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
		allocate();
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

	inline this_type &
	operator =(this_type const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s]=that[s];
		});

		return (*this);
	}

	template<typename TR> inline this_type &
	operator =(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s]= domain_.manifold_->calculate(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] +=domain_.calculate(that, s);
		});

		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{
		allocate();
		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] -= domain_.calculate(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] *= domain_.calculate(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] /= domain_.calculate(that, s);
		});

		return (*this);
	}
///@}

	template<typename TFun> void pull_back(TFun const &fun)
	{
		pull_back(domain_, fun);
	}

	template<typename TD, typename TFun> void pull_back(TD const & domain,
			TFun const &fun)
	{
		allocate();

		parallel_for(domain, [&](index_type const & s)
		{

			//FIXME geometry coordinates convert

				(*this)[s] = domain_.sample( s,fun(
								//domain.MapTo(domain_.InvMapTo(
								domain_.coordinates(s)
								//))
						)
				);
			});

	}

	typedef typename std::conditional<
			domain_type::iform == VERTEX || domain_type::iform == VOLUME,
			value_type, nTuple<value_type, 3>>::type field_value_type;

	field_value_type operator()(
			typename domain_type::coordinates_type const& x) const
	{
		return std::move(domain_.gather(*this, x));
	}

	field_value_type gather(typename domain_type::coordinates_type const& x)
	{
		return std::move(domain_.gather(*this, x));

	}

	template<typename ...Args>
	void scatter(Args && ... args)
	{
		domain_.scatter(const_cast<this_type&>(*this),
				std::forward<Args>(args)...);
	}

	template<typename ...Args>
	auto dataset_shape(Args &&... args) const
	DECL_RET_TYPE(( domain_. dataset_shape( std::forward<Args>(args)...)))

}
;

template<typename TC, typename TD>
struct reference_traits<_Field<TC, TD> >
{
	typedef _Field<TC, TD> const & type;
};

template<typename > struct field_result_of;
template<typename ... > struct index_of;

template<typename TC, typename TD, typename TI>
struct index_of<_Field<TC, TD>, TI>
{
	typedef typename _Field<TC, TD>::value_type type;
};

template<typename TOP, typename ...T, typename TI>
struct index_of<_Field<Expression<TOP, T...>>, TI>
{
	typedef typename field_result_of<TOP(T..., TI)>::type type;
};

template<typename TOP, typename TL, typename TI>
struct field_result_of<TOP(TL, TI)>
{
	typedef typename result_of<TOP(typename index_of<TL, TI>::type)>::type type;
};

template<typename TOP, typename TL, typename TR, typename TI>
struct field_result_of<TOP(TL, TR, TI)>
{
	typedef typename result_of<
			TOP(typename index_of<TL, TI>::type,
					typename index_of<TR, TI>::type)>::type type;
};

template<typename TV, typename TDomain> using Field= _Field< std::shared_ptr<TV>,TDomain >;

template<typename > struct is_field
{
	static constexpr bool value = false;
};

template<typename ...T> struct is_field<_Field<T...>>
{
	static constexpr bool value = true;
};

template<typename ... >struct Expression;

template<typename ...> struct field_traits;

template<typename T> struct field_traits<T>
{

	static constexpr size_t ndims = 0;

	static constexpr size_t iform = VERTEX;
};

template<typename TC, typename TD>
struct field_traits<_Field<TC, TD>>
{

	static constexpr size_t ndims = TD::ndims;

	static constexpr size_t iform = TD::iform;
};

/// \defgroup   Field Expression
/// @{

template<typename ...> struct field_traits
{
	typedef std::nullptr_t domain_type;
};

template<typename TOP, typename TL>
struct field_traits<_Field<Expression<TOP, TL> >>
{

	static constexpr size_t ndims = field_traits<TL>::ndims;

	static constexpr size_t iform = field_traits<TL>::iform;

};

template<typename TOP, typename TL, typename TR>
struct field_traits<_Field<Expression<TOP, TL, TR> >>
{

	typedef typename std::conditional<is_field<TL>::value, field_traits<TL>,
			field_traits<TR>>::type traits_type;

	static constexpr size_t ndims = traits_type::ndims;

	static constexpr size_t iform = traits_type::iform;

};
// FIXME just a temporary path, need fix
template<typename TOP, typename TR>
struct field_traits<_Field<Expression<TOP, double, TR> >>
{

	static constexpr size_t ndims = field_traits<TR>::ndims;

	static constexpr size_t iform = field_traits<TR>::iform;

};

template<typename TOP, typename ...TL>
struct _Field<Expression<TOP, TL...>> : public Expression<TOP, TL...>
{
	typedef _Field<Expression<TOP, TL...>> this_type;

	using Expression<TOP, TL...>::Expression;

};

template<typename TOP, typename ...TL>
struct _Field<BooleanExpression<TOP, TL...>> : public Expression<TOP, TL...>
{
	typedef _Field<BooleanExpression<TOP, TL...>> this_type;

	using Expression<TOP, TL...>::Expression;

	operator bool() const
	{
		UNIMPLEMENT;
		return false;
	}
};

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)

template<typename TV, typename ... Others>
auto make_field(Others && ...others)
DECL_RET_TYPE((_Field<std::shared_ptr<TV>,
				typename std::remove_reference<Others>::type...>(
						std::forward<Others>(others)...)))

template<typename, size_t> class Domain;
//
//template<typename TV, size_t IFORM, typename TM, typename ... Others>
//auto make_form(TM const &manifold,
//		Others && ...others)
//				DECL_RET_TYPE((_Field<std::shared_ptr<TV>,Domain<TM,IFORM>>(
//										Domain<TM,IFORM>(manifold),std::forward<Others>(others)...)))

template<typename TV, size_t IFORM, typename TM, typename ... Others>
_Field<std::shared_ptr<TV>, Domain<TM, IFORM>> make_form(
		std::shared_ptr<TM> manifold, Others && ...others)
{
	return std::move(
			_Field<std::shared_ptr<TV>, Domain<TM, IFORM>>(
					Domain<TM, IFORM>(manifold->shared_from_this()),
					std::forward<Others>(others)...));
}

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_H_ */
