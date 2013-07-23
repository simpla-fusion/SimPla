/*
 * field.h
 *
 *  Created on: 2013-7-19
 *      Author: salmon
 */

#ifndef FIELD_H_
#define FIELD_H_
#include <memory> // for shared_ptr
#include <algorithm> //for swap
#include <utility>
#include <type_traits>

namespace simpla
{

template<typename TGeometry, typename TExpr>
struct Field
{
public:
	typedef TGeometry GeometryType;

	typename ConstReferenceTraits<GeometryType>::type geometry;

	typename ReferenceTraits<TExpr>::type expr;

	typedef typename remove_const_reference<decltype(expr[0])>::type ValueType;

	typedef Field<GeometryType, TExpr> ThisType;

	typedef typename TGeometry::CoordinatesType CoordinatesType;

	Field()
	{
	}

	template<typename TG>
	Field(TG const & g) :
			geometry(g), expr(geometry.get_num_of_elements())
	{
	}

	template<typename TG, typename TE>
	Field(TG const & g, TE e) :
			geometry(g), expr(e)
	{
	}

	Field(ThisType const &) = delete;

	void swap(ThisType & rhs)
	{
		GeometryType::swap(rhs);
		TExpr::swap(rhs);
	}

	virtual ~Field()
	{
	}

	inline ThisType & operator=(ThisType const & rhs)
	{
		geometry.grid.Assign(*this,rhs);
		return (*this);
	}

	template<typename TR>
	inline ThisType & operator=(TR const & rhs)
	{
		geometry.grid.Assign(*this,rhs);
		return (*this);
	}

	inline ValueType Get(CoordinatesType const &x,Real effect_radius=0)const
	{
		return (geometry.IntepolateFrom(expr,x,effect_radius));
	}

	inline void Put(ValueType const & v,CoordinatesType const &x,Real effect_radius=0)
	{
		geometry.IntepolateTo(expr,v,x,effect_radius);
	}
	template<typename TIDX>
	inline ValueType & operator[](TIDX const &s )
	{
		return (expr[s]);
	}
	template<typename TIDX>
	inline ValueType const & operator[](TIDX const &s )const
	{
		return (expr[s]);
	}
};

template<typename T,typename INDEX>
auto eval(T const &f ,INDEX const & s)
->typename std::enable_if<is_Field<T>::value,decltype(get_grid(f).eval(f,s))>::type
{
	return (get_grid(f).eval(f,s));
}

template<typename TG,typename TE> auto get_grid(Field<TG, TE> const & f)
DECL_RET_TYPE(f.geometry.grid)

template<typename TOP, typename TExpr> auto get_grid(
		UniOp<TOP, TExpr> const & f) ->
		typename std::enable_if<is_Field<TExpr>::value,
		typename remove_const_reference<decltype(get_grid(f.expr))>::type>::type const &
{
	return (get_grid(f.expr));
}

template<typename TOP, typename TL, typename TR> auto get_grid(
		BiOp<TOP, TL, TR> const & f)
		-> typename std::enable_if<is_Field<TL>::value,
		typename remove_const_reference<decltype(get_grid(f.l_.expr))>::type>::type const &
{
	return (get_grid(f.l_.expr));
}
template<typename TOP, typename TL, typename TR> auto get_grid(
		BiOp<TOP, TL, TR> const & f)
		-> typename std::enable_if<(!is_Field<TL>::value) && is_Field<TR>::value,
		typename remove_const_reference<decltype(get_grid(f.r_.expr))>::type>::type const &
{
	return (get_grid(f.r_.expr));
}
template<typename T> struct order_of_form
{
	static const int value = -10000;
};
template<typename TG, int IFORM, typename TE>
struct order_of_form<Field<Geometry<TG, IFORM>, TE> >
{
	static const int value = IFORM;
};

template<typename TOP, typename TF>
struct order_of_form<UniOp<TOP, TF> >
{
	static const int value = order_of_form<TF>::value;
};

template<typename TOP, typename TL, typename TR>
struct order_of_form<BiOp<TOP, TL, TR> >
{
	static const int value = order_of_form<TL>::value;
};
}
// namespace simpla

#endif /* FIELD_H_ */
