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

template<typename TG, typename TE, typename TOP>
struct Field<TG, BiOp<TOP, TE> > : public BiOp<TOP, TE>
{
public:
	typedef TG GeometryType;

	typename ConstReferenceTraits<GeometryType>::type geometry;

	typedef Field<TG, BiOp<TOP, TE> > ThisType;

	Field(TE const & l) :
			geometry(l.geometry.grid), BiOp<TOP, TE>(l)
	{
	}

	Field(ThisType const &) =default;

	template<typename IDX> inline auto operator[](IDX const & idx) const
	->typename remove_const_reference<decltype(geometry.eval(*this,idx))>::type
	{
		return ((geometry.eval(*this,idx)));
	}
};

template<typename TG, typename TL, typename TR, typename TOP>
struct Field<TG, BiOp<TOP, TL, TR> > : public BiOp<TOP, TL, TR>
{
public:
	typedef TG GeometryType;

	typename ConstReferenceTraits<GeometryType>::type geometry;

	typedef Field<TG, BiOp<TOP, TL, TR> > ThisType;

	Field(TL const & l, TR const & r) :
			geometry(l.geometry.grid), BiOp<TOP, TL, TR>(l, r)
	{
	}

	Field(ThisType const &) =default;

	template<typename IDX>
	inline auto operator[](IDX const & idx) const
	->typename remove_const_reference<decltype(geometry.eval(*this,idx))>::type
	{
		return (geometry.eval(*this,idx));
	}

};
template<typename TG>
struct Field<TG, Zero>
{
public:

	typedef Field<TG, Zero> ThisType;

	Field()
	{
	}

	Field(ThisType const &) =default;

	template<typename IDX>
	inline Zero operator[](IDX const & ) const
	{
		return (Zero());
	}

};

}
// namespace simpla

#endif /* FIELD_H_ */
