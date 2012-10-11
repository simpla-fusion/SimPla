/*
 * vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_
#include "fetl_defs.h"
#include "ntuple.h"

namespace simpla
{
namespace fetl
{
namespace vector_calculus
{

template<typename TL, typename TR> struct OpDot;
template<typename TL, typename TR> struct OpCross;

template<typename TL> struct OpGrad;
template<typename TL> struct OpDiverge;
template<typename TL> struct OpCurl;
template<int IR, typename TL> struct OpCurlPD;
} // namespace vector_calculus

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
		vector_calculus::OpDot<Field<IZeroForm, nTuple<N, TVL>, TLExpr>,
				Field<IZeroForm, nTuple<N, TVR>, TRExpr> > >                  //
Dot(Field<IZeroForm, nTuple<N, TVL>, TLExpr> const & lhs,
		Field<IZeroForm, nTuple<N, TVR>, TRExpr> const & rhs)
{
	return (Field<IZeroForm,
			typename arithmetic::OpMultiplication<TVL, TVR>::ValueType,
			vector_calculus::OpDot<Field<IZeroForm, nTuple<N, TVL>, TLExpr>,
					Field<IZeroForm, nTuple<N, TVR>, TRExpr> > >(lhs, rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		typename vector_calculus::OpDot<nTuple<N, TVL, TLExpr>, nTuple<N, TVR> >::ValueType,
		vector_calculus::OpDot<nTuple<N, TVL, TLExpr>,
				Field<IZeroForm, nTuple<N, TVR>, TRExpr> > >                  //
Dot(nTuple<N, TVL, TLExpr> const & lhs,
		Field<IZeroForm, nTuple<N, TVR>, TRExpr> const &rhs)
{

	return (Field<IZeroForm,
			typename vector_calculus::OpDot<nTuple<N, TVL, TLExpr>,
					nTuple<N, TVR> >::ValueType,
			vector_calculus::OpDot<nTuple<N, TVL, TLExpr>,
					Field<IZeroForm, nTuple<N, TVR>, TRExpr> > >(lhs, rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		typename vector_calculus::OpDot<nTuple<N, TVL, TLExpr>, nTuple<N, TVR> >::ValueType,
		vector_calculus::OpDot<Field<IZeroForm, nTuple<N, TVL>, TLExpr>,
				nTuple<N, TVR, TRExpr> > >                                    //
Dot(Field<IZeroForm, nTuple<N, TVL>, TLExpr> const & lhs,
		nTuple<N, TVR, TRExpr> const & rhs)
{

	return (Field<IZeroForm,
			typename vector_calculus::OpDot<nTuple<N, TVL, TLExpr>,
					nTuple<N, TVR> >::ValueType,
			vector_calculus::OpDot<Field<IZeroForm, nTuple<N, TVL>, TLExpr>,
					nTuple<N, TVR, TRExpr> > >(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
		vector_calculus::OpCross<Field<IZeroForm, nTuple<THREE, TVL>, TLExpr>,
				Field<IZeroForm, nTuple<THREE, TVR>, TRExpr> > >              //
Cross(Field<IZeroForm, nTuple<THREE, TVL, NullType>, TLExpr> const & lhs,
		Field<IZeroForm, nTuple<THREE, TVR, NullType>, TRExpr> const & rhs)
{
	return (Field<IZeroForm,
			nTuple<THREE,
					typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
			vector_calculus::OpCross<
					Field<IZeroForm, nTuple<THREE, TVL>, TLExpr>,
					Field<IZeroForm, nTuple<THREE, TVR>, TRExpr> > >(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
		vector_calculus::OpCross<nTuple<THREE, TVL, TLExpr>,
				Field<IZeroForm, nTuple<THREE, TVR>, TRExpr> > >              //
Cross(nTuple<THREE, TVL, TLExpr> const & lhs,
		Field<IZeroForm, nTuple<THREE, TVR, NullType>, TRExpr> const &rhs)
{
	return (Field<IZeroForm,
			nTuple<THREE,
					typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
			vector_calculus::OpCross<nTuple<THREE, TVL, TLExpr>,
					Field<IZeroForm, nTuple<THREE, TVR, NullType>, TRExpr> > >(
			lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline Field<IZeroForm,
		nTuple<THREE, typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
		vector_calculus::OpCross<
				Field<IZeroForm, nTuple<THREE, TVL, NullType>, TLExpr>,
				nTuple<THREE, TVR, TRExpr> > >                                //
Cross(Field<IZeroForm, nTuple<THREE, TVL, NullType>, TLExpr> const & lhs,
		nTuple<THREE, TVR, TRExpr> const & rhs)
{
	return (Field<IZeroForm,
			nTuple<THREE,
					typename arithmetic::OpMultiplication<TVL, TVR>::ValueType>,
			vector_calculus::OpCross<
					Field<IZeroForm, nTuple<THREE, TVL, NullType>, TLExpr>,
					nTuple<THREE, TVR, TRExpr> > >(lhs, rhs));
}

template<typename TVL, typename TLExpr>
struct Field<IOneForm, TVL,
		vector_calculus::OpGrad<Field<IZeroForm, TVL, TLExpr> > >
{

	typedef Field<IZeroForm, TVL, TLExpr> TL;
	typename TypeTraits<TL>::ConstReference lhs_;

	static const int IForm = IOneForm;

	typedef TVL ValueType;

	typedef Field<IForm, ValueType, vector_calculus::OpGrad<TL> > ThisType;

	typedef typename TL::Grid Grid;

	Grid const & grid; // FIXME need grid detriment

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (grid.grad_(lhs_, s));
	}
};

template<typename TVL, typename TLExpr>
inline Field<IOneForm, TVL,
		vector_calculus::OpGrad<Field<IZeroForm, TVL, TLExpr> > >             //
Grad(Field<IZeroForm, TVL, TLExpr> const & lhs)
{
	return (Field<IOneForm, TVL,
			vector_calculus::OpGrad<Field<IZeroForm, TVL, TLExpr> > >(lhs));
}

template<typename TVL, typename TLExpr>
struct Field<IZeroForm, TVL,
		vector_calculus::OpDiverge<Field<IOneForm, TVL, TLExpr> > >
{

	typedef Field<IOneForm, TVL, TLExpr> TL;
	typename TypeTraits<TL>::ConstReference lhs_;

	static const int IForm = IZeroForm;

	typedef TVL ValueType;

	typedef typename TL::Grid Grid;

	typedef Field<IForm, ValueType, vector_calculus::OpDiverge<TL> > ThisType;

	Grid const & grid; // FIXME need grid detriment

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (grid.diverge_(lhs_, s));
	}

};

template<typename TVL, typename TLExpr>
inline Field<IZeroForm, TVL,
		vector_calculus::OpDiverge<Field<IOneForm, TVL, TLExpr> > >           //
Diverge(Field<IOneForm, TVL, TLExpr> const & lhs)
{
	return (Field<IZeroForm, TVL,
			vector_calculus::OpDiverge<Field<IOneForm, TVL, TLExpr> > >(lhs));
}
template<typename TVL, typename TLExpr>
struct Field<ITwoForm, TVL,
		vector_calculus::OpCurl<Field<IOneForm, TVL, TLExpr> > >
{
	typedef Field<IOneForm, TVL, TLExpr> TL;
	typename TypeTraits<TL>::ConstReference lhs_;

	static const int IForm = ITwoForm;

	typedef TVL ValueType;

	typedef Field<IForm, ValueType, vector_calculus::OpCurl<TL> > ThisType;

	typedef typename TL::Grid Grid;

	Grid const & grid; // FIXME need grid detriment

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (grid.curl_(lhs_, s));
	}

};

template<typename TVL, typename TLExpr>
inline Field<ITwoForm, TVL,
		vector_calculus::OpCurl<Field<IOneForm, TVL, TLExpr> > >              //
Curl(Field<IOneForm, TVL, TLExpr> const & lhs)
{
	return (Field<ITwoForm, TVL,
			vector_calculus::OpCurl<Field<IOneForm, TVL, TLExpr> > >(lhs));
}

template<typename TVL, typename TLExpr>
struct Field<IOneForm, TVL,
		vector_calculus::OpCurl<Field<ITwoForm, TVL, TLExpr> > >
{
	typedef Field<ITwoForm, TVL, TLExpr> TL;
	typename TypeTraits<TL>::ConstReference lhs_;

	static const int IForm = IOneForm;

	typedef TVL ValueType;

	typedef Field<IForm, ValueType, vector_calculus::OpCurl<TL> > ThisType;

	typedef typename TL::Grid Grid;

	Grid const & grid; // FIXME need grid detriment

	Field(TL const &lhs) :
			grid(lhs.grid), lhs_(lhs)
	{
	}

	inline ValueType operator[](size_t s) const
	{
		return (grid.curl_(lhs_, s));
	}

};

template<typename TVL, typename TLExpr>
inline Field<IOneForm, TVL,
		vector_calculus::OpCurl<Field<ITwoForm, TVL, TLExpr> > >              //
Curl(Field<ITwoForm, TVL, TLExpr> const & lhs)
{
	return (Field<IOneForm, TVL,
			vector_calculus::OpCurl<Field<ITwoForm, TVL, TLExpr> > >(lhs));
}
//
//template<int IPD, typename TVL, typename TLExpr>
//struct UniOp<Field<IOneForm, TVL, TLExpr>, vector_calculus::OpCurlPD<IPD> >
//{
//	typedef Field<IOneForm, TVL, TLExpr> TL;
//	typedef TL LReference;
//	typedef UniOp<TL, vector_calculus::OpCurlPD<IPD> > ThisType;
//	static const int IForm = ITwoForm;
//	typedef TVL ValueType;
//	typedef typename Field<IOneForm, TVL, TLExpr>::Grid Grid;
//	typedef Field<IForm, ValueType, ThisType> ResultType;
//
//	static Grid const & get_grid(TL const & lhs)
//	{
//		return (lhs.grid);
//	}
//	static ValueType op(TL const & lhs, size_t const & s)
//	{
//		return (lhs.grid.template curlPd_<IPD>(lhs, s));
//	}
//};
//template<int IPD, typename TVL, typename TL>
//inline typename UniOp<Field<IOneForm, TVL, TL>, vector_calculus::OpCurlPD<IPD> >::ResultType //
//CurlPD(Field<IOneForm, TVL, TL> const & lhs)
//{
//	typedef typename UniOp<Field<IOneForm, TVL, TL>,
//			vector_calculus::OpCurlPD<IPD> >::ResultType ResultType;
//	return (ResultType(lhs));
//}
//
//template<int IPD, typename TVL, typename TLExpr>
//struct UniOp<Field<ITwoForm, TVL, TLExpr>, vector_calculus::OpCurlPD<IPD> >
//{
//
//	typedef Field<ITwoForm, TVL, TLExpr> TL;
//	typedef TL LReference;
//	typedef UniOp<Field<ITwoForm, TVL, TLExpr>, vector_calculus::OpCurlPD<IPD> > ThisType;
//	static const int IForm = IOneForm;
//	typedef TVL ValueType;
//	typedef typename Field<ITwoForm, TVL, TLExpr>::Grid Grid;
//	typedef Field<IForm, ValueType, ThisType> ResultType;
//
//	static Grid const & get_grid(TL const & lhs)
//	{
//		return (lhs.grid);
//	}
//	static ValueType op(TL const & lhs, size_t const & s)
//	{
//		return (lhs.grid.template curlPd_<IPD>(lhs, s));
//	}
//};
//
//template<int IPD, typename TVL, typename TL>
//inline typename UniOp<Field<ITwoForm, TVL, TL>, vector_calculus::OpCurlPD<IPD> >::ResultType //
//CurlPD(Field<ITwoForm, TVL, TL> const & lhs)
//{
//	typedef typename UniOp<Field<ITwoForm, TVL, TL>,
//			vector_calculus::OpCurlPD<IPD> >::ResultType ResultType;
//	return (ResultType(lhs));
//}

// Vector operation -----------------------------------------
//
////-----------------------------------------
//// 微分算符
//
//template<typename TL>
//class Field<vector_calculus::OpGrad<TL> >
//{
//public:
//	DEFINE_UNIOP_HEAD
//	static const int IForm =
//			(TL::IForm == IZeroForm) ? (IOneForm) : (INullForm);
//
//	template<typename TI>
//	inline ValueType operator[](TI const &s) const
//	{
//		return (grid.grad_(Int2Type<TL::IForm>(), *this, s));
//
//	}
//};
//
//template<typename TL>
//class Field<vector_calculus::OpDiverge<TL> >
//{
//public:
//
//	DEFINE_UNIOP_HEAD
//
//	static const int IForm =
//			(TL::IForm == IOneForm) ? (IZeroForm) : (INullForm);
//
//	template<typename TI>
//	inline ValueType operator[](TI const &s) const
//	{
//		return (grid.diverge_(Int2Type<TL::IForm>(), *this, s));
//	}
//};
//
//template<typename TL>
//struct Field<vector_calculus::OpCurl<TL> >
//{
//public:
//	DEFINE_UNIOP_HEAD
//
//	static const int IForm =
//			(TL::IForm == IOneForm) ?
//					(ITwoForm) :
//					((TL::IForm == ITwoForm) ? (IOneForm) : (INullForm));
//
//	template<typename TI>
//	inline ValueType operator[](TI const &s) const
//	{
//		return (grid.curl_(Int2Type<TL::IForm>(), *this, s));
//	}
//};
//
//template<typename TL, int IR>
//struct Field<vector_calculus::OpCurlPD<TL, IR> >
//{
//public:
//	DEFINE_UNIOP_HEAD
//
//	static const int IForm =
//			(TL::IForm == IOneForm) ?
//					(ITwoForm) :
//					((TL::IForm == ITwoForm) ? (IOneForm) : (INullForm));
//
//	template<typename TI>
//	inline ValueType operator[](TI const &s) const
//	{
//		return (grid.curlPd_(Int2Type<IR>(), Int2Type<TL::IForm>(), lhs_, s));
//	}
//};

}// namespace fetl
} // namespace simpla
#endif /* VECTOR_CALCULUS_H_ */
