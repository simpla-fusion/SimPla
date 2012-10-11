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

template<int IFORM, typename TValue, typename TExpr> struct Field;

template<typename TL, typename TR>
struct FieldEquation
{
public:

	typename TypeTraits<TL>::ConstReference lhs_;
	typename TypeTraits<TR>::ConstReference rhs_;

	FieldEquation(TL const & lhs, TR const & rhs) :
			lhs_(lhs), rhs_(rhs)
	{
	}

};

template<int IFORM, typename TValue, typename TLExpr, typename TRExpr>
FieldEquation<Field<IFORM, TValue, TLExpr>, Field<IFORM, TValue, TRExpr> >    //
operator==(Field<IFORM, TValue, TLExpr> const & lhs,
		Field<IFORM, TValue, TRExpr> const & rhs)
{
	return (FieldEquation<Field<IFORM, TValue, TLExpr>,
			Field<IFORM, TValue, TRExpr> >(lhs, rhs));
}

//template<int I, typename T, typename TL, typename TOP>
//class Field<I, T, UniOp<TL, TOP> >
//{
//public:
//	typedef UniOp<TL, TOP> OpType;
//
//	static const int IForm = OpType::IForm;
//
//	typedef typename UniOp<TL, TOP>::Grid Grid;
//	typedef typename OpType::ValueType ValueType;
//	typedef Field<IForm, ValueType, OpType> ThisType;
//	typedef ThisType ConstReference;
//
//	typename OpType::LReference lhs_;
//
//	Grid const &grid;
//
//	Field(typename OpType::TL const &lhs) :
//			grid(OpType::get_grid(lhs)), lhs_(lhs)
//	{
//	}
//
//	ValueType operator[](size_t s) const
//	{
//		return (OpType::op(lhs_, s));
//	}
//
//};
//
//template<int N, typename T, typename TL, typename TR, typename TOP>
//struct Field<N, T, BiOp<TL, TR, TOP> >
//{
//	typedef BiOp<TL, TR, TOP> OpType;
//
//	static const int IForm = OpType::IForm;
//	typedef typename BiOp<TL, TR, TOP>::Grid Grid;
//	typedef typename OpType::ValueType ValueType;
//	typedef Field<IForm, ValueType, BiOp<TL, TR, TOP> > ThisType;
//	typedef ThisType ConstReference;
//
//	typename OpType::LReference lhs_;
//	typename OpType::RReference rhs_;
//
//	Grid const &grid;
//
//	Field(typename OpType::TL const &lhs, typename OpType::TR const & rhs) :
//			grid(OpType::get_grid(lhs, rhs)), lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//	inline ValueType operator[](size_t s) const
//	{
//		return (OpType::op(lhs_, rhs_, s));
//	}
//
//};

template<int IFORM, typename TVL, typename TLExpr, template<typename > class TOP>
struct Field<IFORM, TVL, TOP<Field<IFORM, TVL, TLExpr> > >
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typename TypeTraits<TL>::ConstReference lhs_;

	typedef Field<IFORM, TVL, TOP<Field<IFORM, TVL, TLExpr> > > ThisType;

	static const int IForm = IFORM;

	typedef typename TOP<TVL>::ValueType ValueType;

	typedef typename TL::Grid Grid;

	Grid const &grid;

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return TOP<TVL>::eval(lhs_[s]);
	}

};

template<typename, typename > struct FieldOpTriats;

template<int IFORML, typename TVL, typename TLExpr, int IFORMR, typename TVR,
		typename TRExpr>
struct FieldOpTriats<Field<IFORML, TVL, TLExpr>, Field<IFORMR, TVR, TRExpr> >
{
	typedef Field<IFORML, TVL, TLExpr> TL;
	typedef Field<IFORMR, TVR, TRExpr> TR;
	typedef TVL LeftValueType;
	typedef TVR RightValueType;

	typedef typename TL::Grid Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return l.grid;
	}
};
template<typename TVL, int IFORMR, typename TVR, typename TRExpr>
struct FieldOpTriats<TVL, Field<IFORMR, TVR, TRExpr> >
{
	typedef TVL TL;
	typedef Field<IFORMR, TVR, TRExpr> TR;

	typedef TVL LeftValueType;
	typedef TVR RightValueType;

	typedef typename TR::Grid Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return r.grid;
	}
};

template<int IFORML, typename TVL, typename TLExpr, typename TVR>
struct FieldOpTriats<Field<IFORML, TVL, TLExpr>, TVR>
{
	typedef Field<IFORML, TVL, TLExpr> TL;
	typedef TVR TR;

	typedef TVL LeftValueType;
	typedef TVR RightValueType;

	typedef typename TL::Grid Grid;

	static inline Grid const &grid(TL const &l, TR const &r)
	{
		return l.grid;
	}
};

template<int IFORM, typename TV, typename TL, typename TR, template<typename,
		typename > class TOP>
struct Field<IFORM, TV, TOP<TL, TR> >
{
	static const int IForm = IFORM;
	typedef TV ValueType;

	typedef Field<IForm, ValueType, TOP<TL, TR> > ThisType;
	typename TypeTraits<TL>::ConstReference lhs_;
	typename TypeTraits<TR>::ConstReference rhs_;

	typedef typename FieldOpTriats<TL, TR>::Grid Grid;
	typedef typename FieldOpTriats<TL, TR>::LeftValueType LeftValueType;
	typedef typename FieldOpTriats<TL, TR>::RightValueType RightValueType;

	Grid const & grid;

	Field(TL const &lhs, TR const & rhs) :
			grid(FieldOpTriats<TL, TR>::grid(lhs, rhs)), lhs_(lhs), rhs_(rhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return TOP<LeftValueType, RightValueType>::eval(index(lhs_, s), index(rhs_, s));
	}

private:
	template<int I, typename V, typename E, typename TINDX>
	static inline V index(Field<I, V, E> const & lhs, TINDX const &s)
	{
		return lhs[s];
	}
	template<typename V, typename TIDX> static inline V index(V const & lhs,
			TIDX const &)
	{
		return lhs;
	}

};

//template<int IFORM, int IFORML, typename TVL, typename TLExpr, int IFORMR,
//		typename TVR, typename TRExpr, template<typename, typename > class TOP>
//struct Field<IFORM, typename TOP<TVL, TVR>::ValueType,
//		TOP<Field<IFORML, TVL, TLExpr>, Field<IFORMR, TVR, TRExpr> > >
//{
//	typedef Field<IFORML, TVL, TLExpr> TL;
//	typedef Field<IFORMR, TVR, TRExpr> TR;
//
//	static const int IForm = IFORM;
//
//	typedef typename TOP<TVL, TVR>::ValueType ValueType;
//
//	typedef Field<IForm, ValueType, TOP<TL, TR> > ThisType;
//
//	typename TypeTraits<TL>::ConstReference lhs_;
//	typename TypeTraits<TR>::ConstReference rhs_;
//
//	typedef typename Field<IFORML, TVL, TLExpr>::Grid Grid;
//
//	Grid const & grid;
//
//	Field(TL const &lhs, TR const & rhs) :
//			grid(lhs.grid), lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//	inline ValueType operator[](size_t s) const
//	{
//		return TOP<TVL, TVR>::eval(lhs_[s], rhs_[s]);
//	}
//
//};
//
//template<int IFORM, int IFORML, typename TVL, typename TLExpr, typename TVR,
//		template<typename, typename > class TOP>
//struct Field<IFORM, typename TOP<TVL, TVR>::ValueType,
//		TOP<Field<IFORML, TVL, TLExpr>, TVR> >
//{
//	typedef Field<IFORML, TVL, TLExpr> TL;
//	typedef TVR TR;
//
//	static const int IForm = IFORM;
//
//	typedef typename TOP<TVL, TVR>::ValueType ValueType;
//
//	typedef Field<IForm, ValueType, TOP<TL, TR> > ThisType;
//
//	typename TypeTraits<TL>::ConstReference lhs_;
//	typename TypeTraits<TR>::ConstReference rhs_;
//
//	typedef typename Field<IFORML, TVL, TLExpr>::Grid Grid;
//
//	Grid const & grid;
//
//	Field(TL const &lhs, TR const & rhs) :
//			grid(lhs.grid), lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//	inline ValueType operator[](size_t s) const
//	{
//		return TOP<TVL, TVR>::eval(lhs_[s], rhs_);
//	}
//
//};
//
//template<int IFORM, typename TVL, int IFORMR, typename TVR, typename TRExpr,
//		template<typename, typename > class TOP>
//struct Field<IFORM, typename TOP<TVL, TVR>::ValueType,
//		TOP<TVL, Field<IFORMR, TVR, TRExpr> > >
//{
//	typedef TVL TL;
//	typedef Field<IFORMR, TVR, TRExpr> TR;
//
//	static const int IForm = IFORM;
//
//	typedef typename TOP<TVL, TVR>::ValueType ValueType;
//
//	typedef Field<IForm, ValueType, TOP<TL, TR> > ThisType;
//
//	typename TypeTraits<TL>::ConstReference lhs_;
//	typename TypeTraits<TR>::ConstReference rhs_;
//
//	typedef typename Field<IFORMR, TVR, TRExpr>::Grid Grid;
//
//	Grid const & grid;
//
//	Field(TL const &lhs, TR const & rhs) :
//			grid(rhs.grid), lhs_(lhs), rhs_(rhs)
//	{
//	}
//
//	inline ValueType operator[](size_t s) const
//	{
//		return TOP<TVL, TVR>::eval(lhs_, rhs_[s]);
//	}
//};
template<typename > struct ElementTypeTraits;

template<typename TV, typename TExpr>
struct ElementTypeTraits<Field<IZeroForm, TV, TExpr> >
{
	typedef TV Type;
};

template<typename TV, typename TExpr>
struct ElementTypeTraits<Field<IOneForm, TV, TExpr> >
{
	typedef nTuple<THREE, TV> Type;
};

template<typename TV, typename TExpr>
struct ElementTypeTraits<Field<ITwoForm, TV, TExpr> >
{
	typedef nTuple<THREE, TV> Type;
};

template<typename TV, typename TExpr>
struct ElementTypeTraits<Field<IThreeForm, TV, TExpr> >
{
	typedef TV Type;
};

template<typename TL, int IR, typename VR, typename TRExpr>
bool CheckEquationHasVariable(TL const & eqn, Field<IR, VR, TRExpr> const & v)
{
	return false;
}
template<int IL, typename VL, typename TLExpr>
bool CheckEquationHasVariable(Field<IL, VL, TLExpr> const & eqn,
		Field<IL, VL, TLExpr> const & v)
{
	return (eqn.IsSame(v));
}

template<typename TL, int IL, typename VL, int IR, typename VR, typename TRExpr,
		template<typename > class TOP>
bool CheckEquationHasVariable(Field<IL, VL, TOP<TL> > const & eqn,
		Field<IR, VR, TRExpr> const & v)
{
	return CheckEquationHasVariable(eqn.lhs_, v);
}

template<typename TL, typename TR, int IL, typename VL, int IR, typename VR,
		typename TRExpr, template<typename, typename > class TOP>
bool CheckEquationHasVariable(Field<IL, VL, TOP<TL, TR> > const & eqn,
		Field<IR, VR, TRExpr> const & v)
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
