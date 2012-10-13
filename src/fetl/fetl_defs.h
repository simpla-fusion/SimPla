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

namespace simpla
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



template<int IFORM, typename TG, typename TExpr>
TR1::shared_ptr<Field<TG, IFORM, typename Field<TG, IFORM, TExpr>::Grid> > //
DuplicateField(Field<TG, IFORM, TExpr> const f)
{
	typedef typename Field<TG, IFORM, TExpr>::Grid Grid;

	return (TR1::shared_ptr<Field<TG, IFORM, Grid> >(
			new Field<TG, IFORM, Grid>(f.grid)));
}

} // namespace simpla

#endif /* FETL_DEFS_H_ */
