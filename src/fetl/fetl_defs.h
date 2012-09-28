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
#include "primitives/ntuple.h"
#include "primitives/typetraits.h"
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

template<typename, typename, typename TOP> struct BiOp;

template<typename, typename TOP> struct UniOp;

template<int IFORM, typename TValue, typename TExpr> struct Field;

template<typename TL, typename TR>
struct FieldEquation
{
public:

	typename simpla::TypeTraits<TL>::ConstReference lhs_;
	typename simpla::TypeTraits<TR>::ConstReference rhs_;

	FieldEquation(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

};

template<int IFORM, typename TValue, typename TLExpr, typename TRExpr>
FieldEquation<Field<IFORM, TValue, TLExpr> ,Field<IFORM, TValue, TRExpr> >  //
operator==(Field<IFORM, TValue, TLExpr> const & lhs
		,Field<IFORM, TValue, TRExpr> const & rhs)
{
	return (FieldEquation<Field<IFORM, TValue, TLExpr>
			,Field<IFORM, TValue, TRExpr> >(lhs, rhs));
}

template<int I, typename T, typename TL, typename TOP>
class Field<I, T, UniOp<TL, TOP> >
{
public:
	typedef UniOp<TL, TOP> OpType;

	static const int IForm = OpType::IForm;

	typedef typename UniOp<TL, TOP>::Grid Grid;
	typedef typename OpType::ValueType ValueType;
	typedef Field<IForm, ValueType, OpType> ThisType;
	typedef ThisType ConstReference;

	typename OpType::LReference lhs_;

	Grid const &grid;

	Field(typename OpType::TL const &lhs) :
			grid(OpType::get_grid(lhs)), lhs_(lhs)
	{
	}

	ValueType operator[](size_t s) const
	{
		return (OpType::op(lhs_, s));
	}

};

template<int N, typename T, typename TL, typename TR, typename TOP>
struct Field<N, T, BiOp<TL, TR, TOP> >
{
	typedef BiOp<TL, TR, TOP> OpType;

	static const int IForm = OpType::IForm;
	typedef typename BiOp<TL, TR, TOP>::Grid Grid;
	typedef typename OpType::ValueType ValueType;
	typedef Field<IForm, ValueType, BiOp<TL, TR, TOP> > ThisType;
	typedef ThisType ConstReference;

	typename OpType::LReference lhs_;
	typename OpType::RReference rhs_;

	Grid const &grid;

	Field(typename OpType::TL const &lhs, typename OpType::TR const & rhs) :
			grid(OpType::get_grid(lhs, rhs)), lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (OpType::op(lhs_, rhs_, s));
	}

};

template<typename > struct ElementTypeTraits;

template<typename TV, typename TExpr> struct ElementTypeTraits<
		Field<IZeroForm, TV, TExpr> >
{
	typedef TV Type;
};

template<typename TV, typename TExpr> struct ElementTypeTraits<
		Field<IOneForm, TV, TExpr> >
{
	typedef nTuple<THREE, TV> Type;
};

template<typename TV, typename TExpr> struct ElementTypeTraits<
		Field<ITwoForm, TV, TExpr> >
{
	typedef nTuple<THREE, TV> Type;
};

template<typename TV, typename TExpr> struct ElementTypeTraits<
		Field<IThreeForm, TV, TExpr> >
{
	typedef TV Type;
};

template<typename TL, int IR, typename VR, typename TRExpr>
bool CheckEquationHasVariable(TL const & eqn, Field<IR, VR, TRExpr> const & v)
{
	return false;
}
template<int IL, typename VL, typename TLExpr>
bool CheckEquationHasVariable(Field<IL, VL, TLExpr> const & eqn
		, Field<IL, VL, TLExpr> const & v)
{
	return (eqn.IsSame(v));
}

template<typename TL, int IL, typename VL, int IR, typename VR, typename TRExpr,
		typename TOP>
bool CheckEquationHasVariable(Field<IL, VL, UniOp<TL, TOP> > const & eqn
		, Field<IR, VR, TRExpr> const & v)
{
	return CheckEquationHasVariable(eqn.lhs_, v);
}

template<typename TL, typename TR, int IL, typename VL, int IR, typename VR,
		typename TRExpr, typename TOP>
bool CheckEquationHasVariable(Field<IL, VL, BiOp<TL, TR, TOP> > const & eqn
		, Field<IR, VR, TRExpr> const & v)
{
	return CheckEquationHasVariable(eqn.lhs_, v)
			|| CheckEquationHasVariable(eqn.rhs_, v);
}

template<typename > struct FieldTraits;

template<int IFORM, typename TV, typename TExpr>
struct FieldTraits<Field<IFORM, TV, TExpr> >
{
	typedef typename Field<IFORM, TV, TExpr>::Grid Grid;
	typedef Field<IFORM, TV, Grid> FieldType;
};

template<int IFORM, typename TV, typename TExpr>
TR1::shared_ptr<Field<IFORM, TV, typename Field<IFORM, TV, TExpr>::Grid> > //
DuplicateField(Field<IFORM, TV, TExpr> const f)
{
	typedef typename Field<IFORM, TV, TExpr>::Grid Grid;

	return (TR1::shared_ptr<Field<IFORM, TV, Grid> >(
			new Field<IFORM, TV, Grid>(f.grid)));
}

} // namespace fetl
} // namespace simpla

#endif /* FETL_DEFS_H_ */
