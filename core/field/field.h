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
template<typename ... >struct Expression;

template<typename TG, size_t IFORM> class Domain;

//template<typename ...T, typename TI>
//auto get_value(_Field<T...> const & f,
//		TI const &s)->typename field_traits<_Field<T...>>::value_type const&
//{
//	return f[s];
//}
//
//template<typename ...T, typename TI>
//auto get_value(_Field<T...> & f,
//		TI const &s)->typename field_traits<_Field<T...>>::value_type const&
//{
//	return f[s];
//}
//
//template<typename ...T, typename TI>
//auto get_value(_Field<Expression<T...>> const & f,
//		TI const &s)->typename field_traits<_Field<Expression<T...>>>::value_type
//{
//	return f[s];
//}

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
			(*this)[s]= domain_.get(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator +=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] +=domain_.get(that, s);
		});

		return (*this);

	}

	template<typename TR>
	inline this_type & operator -=(TR const &that)
	{
		allocate();
		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] -= domain_.get(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator *=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] *= domain_.get(that, s);
		});

		return (*this);
	}

	template<typename TR>
	inline this_type & operator /=(TR const &that)
	{
		allocate();

		parallel_for(domain_, [&](index_type const & s)
		{
			(*this)[s] /= domain_.get(that, s);
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

//	auto dataset() const
//			DECL_RET_TYPE(std::move(
//							std::tuple_cat(std::make_tuple(data_.get(), DataType::create<value_type>())
//									,domain_.dataset()))
//			)

//	auto dataset_shape() const
//	DECL_RET_TYPE(( domain_.dataset_shape()))

	template<typename ...Args>
	auto dataset_shape(Args &&... args) const
	DECL_RET_TYPE(( domain_. dataset_shape( std::forward<Args>(args)...)))

}
;

template<typename TV, typename TDomain> using Field= _Field< std::shared_ptr<TV>,TDomain >;

template<typename > struct is_field
{
	static constexpr bool value = false;
};

template<typename ...T> struct is_field<_Field<T...>>
{
	static constexpr bool value = true;
};

template<typename ...> struct field_traits;

template<typename TC, typename TD, typename ...Others>
struct field_traits<_Field<TC, TD, Others...>>
{
	typedef _Field<TC, TD, Others...> field_type;

	static constexpr size_t ndims = TD::ndims;

	static constexpr size_t iform = TD::iform;

	typedef typename container_traits<TC>::value_type value_type;

	typedef typename std::conditional<iform == VERTEX || iform == VOLUME,
			value_type, nTuple<value_type, 3>>::type field_value_type;

	typedef TD domain_type;

	typedef typename domain_type::manifold_type manifold_type;

//	static auto get_domain(field_type const &f)
//	DECL_RET_TYPE((f.domain()))

	static auto data(field_type & f)
	DECL_RET_TYPE((f.data()))
};

/// \defgroup   Field Expression
/// @{
template<typename ... >struct Expression;

template<typename ...> struct field_traits
{
	typedef std::nullptr_t domain_type;
};

template<typename TOP, typename TL>
struct field_traits<_Field<Expression<TOP, TL> >>
{

	typedef typename field_traits<TL>::domain_type domain_type;

	typedef typename domain_type::manifold_type manifold_type;

	static constexpr size_t ndims = domain_type::ndims;

	static constexpr size_t iform = domain_type::iform;

//	static auto get_domain(TL const &f)
//	DECL_RET_TYPE((field_traits<TL>::get_domain(f)))
//
//	typedef _Field<Expression<TOP, TL> > field_type;
//	static domain_type get_domain(field_type const &f)
//	{
//		return std::move(field_traits<TL>::get_domain(f.lhs));
//	}

};

template<typename TOP, typename TL, typename TR>
struct field_traits<_Field<Expression<TOP, TL, TR> >>
{

	typedef typename field_traits<TL>::domain_type l_domain_type;

	typedef typename field_traits<TR>::domain_type r_domain_type;

	typedef typename std::conditional<is_field<TL>::value, l_domain_type,
			r_domain_type>::type domain_type;

	typedef typename domain_type::manifold_type manifold_type;

	static constexpr size_t ndims = domain_type::ndims;

	static constexpr size_t iform = domain_type::iform;

//	static domain_type get_domain(TL const &f, TR const &)
//	{
//		return std::move(field_traits<TL>::get_domain(f));
//	}
////	DECL_RET_TYPE((field_traits<TL>::get_domain(f )))
//
//	typedef _Field<Expression<TOP, TL, TR> > field_type;
//	static domain_type get_domain(field_type const &f)
//	{
//		return std::move(field_traits<TL>::get_domain(f.lhs));
//	}
};
// FIXME just a temporary path, need fix
template<typename TOP, typename TR>
struct field_traits<_Field<Expression<TOP, double, TR> >>
{

	typedef typename field_traits<TR>::domain_type domain_type;

	typedef typename domain_type::manifold_type manifold_type;

	static constexpr size_t ndims = domain_type::ndims;

	static constexpr size_t iform = domain_type::iform;

//	static domain_type get_domain(double const &, TR const & f)
//	{
//		return std::move(field_traits<TR>::get_domain(f));
//	}
////	DECL_RET_TYPE((field_traits<TL>::get_domain(f )))
//
//	typedef _Field<Expression<TOP, double, TR> > field_type;
//	static domain_type get_domain(field_type const &f)
//	{
//		return std::move(field_traits<TR>::get_domain(f.rhs));
//	}
};

namespace _impl
{

template<typename TC, typename TD>
struct reference_traits<_Field<TC, TD> >
{
	typedef _Field<TC, TD> const & type;
};

} //namespace _impl

template<typename TOP, typename ...TL>
struct _Field<Expression<TOP, TL...>> : public Expression<TOP, TL...>
{
	typedef _Field<Expression<TOP, TL...>> this_type;

	typedef typename field_traits<this_type>::domain_type domain_type;

	using Expression<TOP, TL...>::Expression;

	operator bool() const
	{
		//		auto d = get_domain(*this);
		//		return   parallel_reduce<bool>(d, _impl::logical_and(), *this);
		return false;
	}

};

//#define _SP_DEFINE__Field_EXPR_BINARY_RIGHT_OPERATOR(_OP_,_NAME_)                                                  \
//	template<typename ...T1,typename  T2> _Field<Expression<_impl::_NAME_,_Field<T1...>,T2>> \
//	operator _OP_(_Field<T1...> && l,T2 &&r)  \
//	{return std::move(_Field<Expression<_impl::_NAME_,_Field<T1...>,T2>>(std::forward<_Field<T1...>>(l),std::forward<T2>(r)));}                  \
//
//
//#define _SP_DEFINE__Field_EXPR_BINARY_OPERATOR(_OP_,_NAME_)                                                  \
//	template<typename ...T1,typename  T2> \
//	_Field<Expression<_impl::_NAME_,_Field<T1...>,T2>>\
//	operator _OP_(_Field<T1...> && l,T2 &&r)  \
//	{return std::move(_Field<Expression<_impl::_NAME_,_Field<T1...>,T2>>(std::forward<_Field<T1...>>(l),std::forward<T2>(r)));}                  \
//	template< typename T1,typename ...T2> \
//	_Field<Expression< _impl::_NAME_,T1,_Field< T2...>>> \
//	operator _OP_(T1 && l, _Field< T2...> &&r)                    \
//	{return std::move(_Field<Expression< _impl::_NAME_,T1,_Field< T2...>>>(std::forward<T1>(l),std::forward<_Field<T2...>>(r)));}                  \
//	template< typename ... T1,typename ...T2> \
//	_Field<Expression< _impl::_NAME_,_Field< T1...>,_Field< T2...>>>\
//	operator _OP_(_Field< T1...> && l,_Field< T2...>  &&r)                    \
//	{return  (_Field<Expression< _impl::_NAME_,_Field< T1...>,_Field< T2...>>>(std::forward<_Field<T1...>>(l),std::forward<_Field<T2...>>(r)));}                  \
//
//
//#define _SP_DEFINE__Field_EXPR_UNARY_OPERATOR(_OP_,_NAME_)                           \
//		template<typename ...T> \
//		_Field<Expression<_impl::_NAME_,_Field<T...> >> \
//		operator _OP_(_Field<T...> &&l)  \
//		{return (_Field<Expression<_impl::_NAME_,_Field<T...> >>(l));}   \
//
//#define _SP_DEFINE__Field_EXPR_BINARY_FUNCTION(_NAME_)                                                  \
//			template<typename ...T1,typename  T2> \
//			_Field<Expression<_impl::_##_NAME_,_Field<T1...>,T2>> \
//			_NAME_(_Field<T1...> && l,T2 &&r)  \
//			{return std::move(_Field<Expression<_impl::_##_NAME_,_Field<T1...>,T2>>(std::forward<_Field<T1...>>(l),std::forward<T2>(r)));}                  \
//			template< typename T1,typename ...T2> \
//			_Field<Expression< _impl::_##_NAME_,T1,_Field< T2...>>> \
//			_NAME_(T1 && l, _Field< T2...>&&r)                    \
//			{return std::move(_Field<Expression< _impl::_##_NAME_,T1,_Field< T2...>>>(std::forward<T1>(l),std::forward<_Field<T2...>>(l)));}                  \
//			template< typename ... T1,typename ...T2> \
//			_Field<Expression< _impl::_##_NAME_,_Field< T1...>,_Field< T2...>>> \
//			_NAME_(_Field< T1...> && l,_Field< T2...>  &&r)                    \
//			{return std::move(_Field<Expression< _impl::_##_NAME_,_Field< T1...>,_Field< T2...>>>(std::forward<_Field<T1...>>(l),std::forward<_Field<T2...>>(r)));}                  \
//
//
//#define _SP_DEFINE__Field_EXPR_UNARY_FUNCTION( _NAME_ )                           \
//		template<typename ...T1> \
//		_Field<Expression<_impl::_##_NAME_,_Field<T1 ...>>> \
//		_NAME_(_Field<T1 ...> &&r)  \
//		{return std::move(_Field<Expression<_impl::_##_NAME_,_Field<T1 ...>>>(std::forward<_Field<T1 ...>>(r)));}   \
//
//
//DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA2(_Field)

DEFINE_EXPRESSOPM_TEMPLATE_BASIC_ALGEBRA(_Field)

template<typename TV, typename ... Others>
auto make_field(Others && ...others)
DECL_RET_TYPE((_Field<std::shared_ptr<TV>,
				typename std::remove_reference<Others>::type...>(
						std::forward<Others>(others)...)))
template<typename TV, size_t IFORM, typename TM, typename ... Others>
auto make_form(TM const &manifold,
		Others && ...others)
				DECL_RET_TYPE((_Field<std::shared_ptr<TV>,Domain<TM,IFORM>>(
										Domain<TM,IFORM>(manifold),std::forward<Others>(others)...)))

}
// namespace simpla

#endif /* CORE_FIELD_FIELD_H_ */
