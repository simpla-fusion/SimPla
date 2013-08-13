/*
 * equation.h
 *
 *  Created on: 2013-7-11
 *      Author: salmon
 */

#ifndef EQUATION_H_
#define EQUATION_H_

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

/**
 * A.x - b=0
 *
 *
 * */
template<typename TL> struct LinearExpression;

template<> struct LinearExpression<NullType>
{
	typedef LinearExpression<NullType> ThisType;

	size_t idx;

	LinearExpression(size_t i) :
			idx(i)
	{
	}
	LinearExpression(ThisType const& rhs) = default;
	~LinearExpression()
	{

	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		try
		{
			v.at(idx) += a;
		} catch (...)
		{
			v.insert(std::make_pair(idx, a));
		}
	}
};

typedef LinearExpression<NullType> PlaceHolder;

struct PlaceHolderGenerator
{
	PlaceHolderGenerator(size_t)
	{
	}
	~PlaceHolderGenerator()
	{
	}

	PlaceHolder operator[](size_t s)
	{
		return (PlaceHolder(s));
	}

};

template<typename TL, typename TR>
struct LinearExpression<
		BiOp<OpPlus, LinearExpression<TL>, LinearExpression<TR> > >
{
	typedef LinearExpression<
			BiOp<OpPlus, LinearExpression<TL>, LinearExpression<TR> > > ThisType;
	typedef LinearExpression<TL> T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a);
		r_.assign(v, b, a);
	}
}
;

template<typename TL, typename TR>
struct LinearExpression<BiOp<OpPlus, TL, LinearExpression<TR> > >
{
	typedef LinearExpression<BiOp<OpPlus, TL, LinearExpression<TR> > > ThisType;
	typedef TL T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		b += l_ * a;
		r_.assign(v, b, a);
	}

};
template<typename TL, typename TR>
struct LinearExpression<BiOp<OpPlus, LinearExpression<TL>, TR> >
{
	typedef LinearExpression<BiOp<OpPlus, LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a);
		b += r_ * a;
	}

};

template<typename TL, typename TR>
struct LinearExpression<
		BiOp<OpMinus, LinearExpression<TL>, LinearExpression<TR> > >
{
	typedef LinearExpression<
			BiOp<OpMinus, LinearExpression<TL>, LinearExpression<TR> > > ThisType;
	typedef LinearExpression<TL> T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a);
		r_.assign(v, b, -a);
	}

};

template<typename TL, typename TR>
struct LinearExpression<BiOp<OpMinus, TL, LinearExpression<TR> > >
{
	typedef LinearExpression<BiOp<OpMinus, TL, LinearExpression<TR> > > ThisType;
	typedef TL T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		b += l_ * a;
		r_.assign(v, b, -a);
	}

};
template<typename TL, typename TR>
struct LinearExpression<BiOp<OpMinus, LinearExpression<TL>, TR> >
{
	typedef LinearExpression<BiOp<OpMinus, LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a);
		b -= r_ * a;
	}

};

template<typename TL>
struct LinearExpression<UniOp<OpNegate, LinearExpression<TL> > >
{
	typedef LinearExpression<UniOp<OpNegate, LinearExpression<TL> > > ThisType;
	typedef LinearExpression<TL> T1;

	T1 l_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l) :
			l_(l)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, -a);
	}

};
template<typename TL, typename TR>
struct LinearExpression<BiOp<OpMultiplies, LinearExpression<TL>, TR> >
{
	typedef LinearExpression<BiOp<OpMultiplies, LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a * r_);
	}

};

template<typename TL, typename TR>
struct LinearExpression<BiOp<OpMultiplies, TL, LinearExpression<TR> > >
{
	typedef LinearExpression<BiOp<OpMultiplies, TL, LinearExpression<TR> > > ThisType;
	typedef TL T1;
	typedef LinearExpression<TR> T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		r_.assign(v, b, l_ * a);
	}

};

template<typename TL, typename TR>
struct LinearExpression<BiOp<OpDivides, LinearExpression<TL>, TR> >
{
	typedef LinearExpression<BiOp<OpDivides, LinearExpression<TL>, TR> > ThisType;
	typedef LinearExpression<TL> T1;
	typedef TR T2;

	T1 l_;
	T2 r_;

	LinearExpression(ThisType const& rhs) = default;

	LinearExpression(T1 const & l, T2 const & r) :
			l_(l), r_(r)
	{
	}

	template<typename TV, typename TB>
	inline void assign(TV & v, TB &b) const
	{
		assign(v, b, 1);
	}

	template<typename TV, typename TB, typename TA>
	inline void assign(TV & v, TB &b, TA const & a) const
	{
		l_.assign(v, b, a / r_);
	}

};

template<typename TL, typename TR> inline auto operator*(TL const &lhs,
		LinearExpression<TR> const& rhs)
				DECL_RET_TYPE((LinearExpression<BiOp<OpMultiplies,TL,LinearExpression<TR > > >(lhs ,rhs)))

template<typename TL, typename TR> inline auto operator*(
		LinearExpression<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearExpression<BiOp<OpMultiplies, LinearExpression<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> inline auto operator/(
		LinearExpression<TL> const& lhs,
		TR const& rhs)
				DECL_RET_TYPE((LinearExpression<BiOp<OpDivides, LinearExpression<TL >,TR > >(lhs ,rhs)))

template<typename TL, typename TR> void operator*(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r) = delete;

template<typename TL, typename TR> void operator/(LinearExpression<TL> const& l,
		LinearExpression<TR> const& r) = delete;

template<typename TL>
inline auto operator-(
		LinearExpression<TL> const & expr)
				DECL_RET_TYPE(( LinearExpression<UniOp<OpNegate,LinearExpression<TL> > >(expr)))

template<typename TL, typename TR> inline auto operator+(
		LinearExpression<TL> const& l, LinearExpression<TR> const& r)
		DECL_RET_TYPE(( LinearExpression<BiOp<OpPlus,LinearExpression<TL>,
						LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator+(TL const& l,
		LinearExpression<TR> const& r)
				DECL_RET_TYPE(( LinearExpression<BiOp<OpPlus,TL, LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator+(
		LinearExpression<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearExpression<BiOp<OpPlus,LinearExpression<TL>, TR > > (l,r)))

template<typename TL, typename TR> inline auto operator-(
		LinearExpression<TL> const& l, LinearExpression<TR> const& r)
		DECL_RET_TYPE(( LinearExpression<BiOp<OpMinus,LinearExpression<TL>,
						LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator-(TL const& l,
		LinearExpression<TR> const& r)
				DECL_RET_TYPE(( LinearExpression<BiOp<OpMinus,TL, LinearExpression< TR > > > (l,r)))

template<typename TL, typename TR> inline auto operator-(
		LinearExpression<TL> const& l,
		TR const& r)
				DECL_RET_TYPE(( LinearExpression<BiOp<OpMinus,LinearExpression<TL>, TR > > (l,r)))

template<typename TL, typename TR> inline auto operator==(
		LinearExpression<TL> const& l, LinearExpression<TR> const& r)
		DECL_RET_TYPE((l-r))

template<typename TL, typename TR> inline auto operator==(TL const& l,
		LinearExpression<TR> const& r) DECL_RET_TYPE(l-r)

template<typename TL, typename TR> inline auto operator==(
		LinearExpression<TL> const& l, TR const& r) DECL_RET_TYPE((l-r))

}
// namespace simpla

#endif /* EQUATION_H_ */
