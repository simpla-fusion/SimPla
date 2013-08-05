/*
 * equation.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef EQUATION_H_
#define EQUATION_H_

#include <map>
#include <utility>
#include <map>
#include <tuple>
#include "expression.h"
namespace simpla
{

//template<typename > struct has_Unknown
//{
//	static const bool value = false;
//};
//
//template<> struct has_Unknown<PlaceHolder>
//{
//	static const bool value = true;
//};
//
//template<typename TL, typename TR, template<typename, typename > class TOP> struct has_Unknown<
//		TOP<TL, TR> >
//{
//	static const bool value = has_Unknown<TL>::value && has_Unknown<TR>::value;
//};
//
//template<typename TL, template<typename > class TOP> struct has_Unknown<TOP<TL> >
//{
//	static const bool value = has_Unknown<TL>::value;
//};

template<typename TL> struct LinearExpression
{
	typedef LinearExpression<TL> ThisType;

	size_t idx;
	TL value;

	LinearExpression(size_t i, TL a = 1) :
			idx(i), value(a)
	{
	}
	LinearExpression(ThisType const& rhs) = default;
	~LinearExpression()
	{

	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}

	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		try
		{
			v.at(idx) += value * a;
		} catch (...)
		{
			v.insert(std::make_pair(idx, value * a));
		}
	}

//	template<typename TR> auto operator*(TR const & rhs) const
//	DECL_RET_TYPE((LinearExpression<decltype( value*rhs) >( idx, value*rhs)))
//
//	template<typename TR> auto operator/(TR const & rhs) const
//	DECL_RET_TYPE((LinearExpression<decltype( value/rhs) >( idx, value/rhs)))

};

typedef LinearExpression<double> PlaceHolder;

template<typename TL, typename TR>
struct LinearExpression<OpPlus<LinearExpression<TL>, LinearExpression<TR> > >
{
	typedef LinearExpression<OpPlus<LinearExpression<TL>, LinearExpression<TR> > > ThisType;
	typedef LinearExpression<TL> T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}

	template<typename TV, typename T>
	void assign(TV & v, T const & a) const
	{
		l_.assign(v, a);
		r_.assign(v, a);
	}

};

template<typename TL, typename TR>
struct LinearExpression<OpMinus<LinearExpression<TL>, LinearExpression<TR> > >
{
	typedef LinearExpression<
			OpMinus<LinearExpression<TL>, LinearExpression<TR> > > ThisType;
	typedef LinearExpression<TL> T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}
	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		l_.assign(v, a);
		r_.assign(v, -a);
	}

};

template<typename TL>
struct LinearExpression<OpNegate<LinearExpression<TL> > >
{
	typedef LinearExpression<OpNegate<LinearExpression<TL> > > ThisType;
	typedef LinearExpression<TL> T1;

	T1 l_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l) :
			l_(l)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}
	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		l_.assign(v, -a);
	}

};
template<typename TL, typename TR>
struct LinearExpression<OpMultiplies<LinearExpression<TL>, TR> >
{
	typedef LinearExpression<OpMultiplies<LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}
	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		l_.assign(v, a * r_);
	}

};

template<typename TL, typename TR>
struct LinearExpression<OpMultiplies<TL, LinearExpression<TR> > >
{
	typedef LinearExpression<OpMultiplies<TL, LinearExpression<TR> > > ThisType;
	typedef TL T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}
	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		r_.assign(v, l_ * a);
	}

};

template<typename TL, typename TR>
struct LinearExpression<OpDivides<LinearExpression<TL>, TR> >
{
	typedef LinearExpression<OpDivides<LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV>
	inline void assign(TV & v) const
	{
		assign(v, 1);
	}
	template<typename TV, typename T>
	inline void assign(TV & v, T const & a) const
	{
		l_.assign(v, a / r_);
	}

};

template<typename TL, typename TR> auto operator*(TL const &lhs,
		LinearExpression<TR> const& rhs)
				DECL_RET_TYPE((LinearExpression<OpMultiplies<TL,LinearExpression<TR > > >(lhs ,rhs)))

template<typename TL, typename TR> auto operator*(
		LinearExpression<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearExpression<OpMultiplies< LinearExpression<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> auto operator/(
		LinearExpression<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearExpression<OpDivides< LinearExpression<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> void operator*(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r) = delete;

template<typename TL, typename TR> void operator/(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r) = delete;

template<typename TL>
auto operator-(LinearExpression<TL> const & expr)
DECL_RET_TYPE(( LinearExpression<OpNegate<LinearExpression<TL> > >(expr)))

template<typename TL, typename TR> auto operator+(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r)
		DECL_RET_TYPE(( LinearExpression<OpPlus<LinearExpression<TL>,
						LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> auto operator+(TL const& l,
		LinearExpression<TR> const& r)
				DECL_RET_TYPE(( LinearExpression<OpPlus<LinearExpression<TL>,
								LinearExpression< TR > > > (LinearExpression<TL>(-1,l),r)))

template<typename TL, typename TR> auto operator+(LinearExpression<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearExpression<OpPlus<LinearExpression<TL>,
								LinearExpression< TR > > > (l,(LinearExpression< TR >(-1,r)))))

template<typename TL, typename TR> auto operator-(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r)
		DECL_RET_TYPE(( LinearExpression<OpMinus<LinearExpression<TL>,
						LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> auto operator--(TL const& l,
		LinearExpression<TR> const& r)
				DECL_RET_TYPE(( LinearExpression<OpMinus<LinearExpression<TL>,
								LinearExpression< TR > > > (LinearExpression<TL>(-1,l),r)))

template<typename TL, typename TR> auto operator-(LinearExpression<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearExpression<OpMinus<LinearExpression<TL>,
								LinearExpression< TR > > > (l,(LinearExpression< TR >(-1,r)))))

}
// namespace simpla

#endif /* EQUATION_H_ */
