/*
 * fetl_defs.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */

#ifndef FETL_DEFS_H_
#define FETL_DEFS_H_
#include "ntuple.h"

namespace simpla
{
namespace fetl
{

enum TopologyID
{
	INullForm = -1,

	IZeroForm = 0, IOneForm = 1, ITwoForm = 2, IThreeForm = 3, IFourForm = 4
}
;

template<typename TG, int IFORM, typename TExpr> struct Field;
//template<typename TL, typename TR>
//struct FieldEquation
//{
//public:
//
//	typename TypeTraits<TL>::ConstReference lhs_;
//	typename TypeTraits<TR>::ConstReference rhs_;
//
//	FieldEquation(TL const & lhs, TR const & rhs) :
//			lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//};
//
//template<typename TG, int IFORM, typename TLExpr, typename TRExpr>
//FieldEquation<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> >            //
//operator==(Field<TG, IFORM, TLExpr> const & lhs,
//		Field<TG, IFORM, TRExpr> const & rhs)
//{
//	return (FieldEquation<Field<TG, IFORM, TLExpr>, Field<TG, IFORM, TRExpr> >(
//			lhs, rhs));
//}

template<typename, typename > struct FieldOpTriats;

template<typename TG, int IFORML, typename TLExpr, int IFORMR, typename TRExpr>
struct FieldOpTriats<Field<TG, IFORML, TLExpr>, Field<TG, IFORMR, TRExpr> >
{
	typedef Field<TG, IFORML, TLExpr> TL;
	typedef Field<TG, IFORMR, TRExpr> TR;
	typedef typename Field<TG, IFORML, TLExpr>::ValueType LeftValueType;
	typedef typename Field<TG, IFORMR, TRExpr>::ValueType RightValueType;

	typedef TG Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return l.grid;
	}
};
template<typename TVL, typename TG, int IFORMR, typename TRExpr>
struct FieldOpTriats<TVL, Field<TG, IFORMR, TRExpr> >
{
	typedef TVL TL;
	typedef Field<TG, IFORMR, TRExpr> TR;

	typedef TVL LeftValueType;
	typedef typename Field<TG, IFORMR, TRExpr>::ValueType RightValueType;

	typedef typename TR::Grid Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return r.grid;
	}
};

template<typename TG, int IFORML, typename TLExpr, typename TVR>
struct FieldOpTriats<Field<TG, IFORML, TLExpr>, TVR>
{
	typedef Field<TG, IFORML, TLExpr> TL;
	typedef TVR TR;

	typedef typename Field<TG, IFORML, TLExpr>::ValueType LeftValueType;
	typedef TVR RightValueType;

	typedef typename TL::Grid Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return l.grid;
	}
};

template<typename TG, typename TL, int IR, typename TRExpr>
bool CheckEquationHasVariable(TL const & eqn, Field<TG, IR, TRExpr> const & v)
{
	return false;
}
template<typename TG, int IL, typename TLExpr, typename TRExpr>
bool CheckEquationHasVariable(Field<TG, IL, TLExpr> const & eqn,
		Field<TG, IL, TRExpr> const & v)
{
	return (eqn.IsSame(v));
}

template<typename TG, typename TLExpr, int IL, int IR, typename TRExpr,
		template<typename > class TOP>
bool CheckEquationHasVariable(Field<TG, IL, TOP<TLExpr> > const & eqn,
		Field<TG, IR, TRExpr> const & v)
{
	return CheckEquationHasVariable(eqn.lhs_, v);
}

template<typename TG, typename TL, typename TR, int IL, int IR, typename TRExpr,
		template<typename, typename > class TOP>
bool CheckEquationHasVariable(Field<TG, IL, TOP<TL, TR> > const & eqn,
		Field<TG, IR, TRExpr> const & v)
{
	return CheckEquationHasVariable(eqn.lhs_, v)
			|| CheckEquationHasVariable(eqn.rhs_, v);
}

template<typename > struct FieldTraits;

template<int IFORM, typename TG, typename TExpr>
struct FieldTraits<Field<TG, IFORM, TExpr> >
{
	typedef typename Field<TG, IFORM, TExpr>::Grid Grid;
	typedef Field<TG, IFORM, Grid> FieldType;
};

template<int IFORM, typename TG, typename TExpr>
TR1::shared_ptr<Field<TG, IFORM, typename Field<TG, IFORM, TExpr>::Grid> > //
DuplicateField(Field<TG, IFORM, TExpr> const f)
{
	typedef typename Field<TG, IFORM, TExpr>::Grid Grid;

	return (TR1::shared_ptr<Field<TG, IFORM, Grid> >(
			new Field<TG, IFORM, Grid>(f.grid)));
}

} // namespace fetl

} // namespace simpla

#endif /* FETL_DEFS_H_ */
