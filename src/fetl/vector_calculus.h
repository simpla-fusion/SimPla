/*
 * vector_calculus.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 */

#ifndef VECTOR_CALCULUS_H_
#define VECTOR_CALCULUS_H_
#include "fetl/fetl_defs.h"
namespace simpla
{
namespace fetl
{
template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, nTuple<N, TVL> , TLExpr>
		,Field<IZeroForm, nTuple<N, TVR> , TRExpr> ,vector_calculus::OpDot>
{

	typedef Field<IZeroForm, nTuple<N, TVL> , TLExpr> TL;
	typedef Field<IZeroForm, nTuple<N, TVR> , TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, vector_calculus::OpDot> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<N, TVL> , nTuple<N, TVR>
			, vector_calculus::OpDot>::ValueType ValueType;
	typedef typename TL::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Dot(lhs[s], rhs[s]));
	}
};

template<typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TL>
		,Field<IZeroForm, nTuple<THREE, TVR> , TR> ,vector_calculus::OpDot>::ResultType //
Dot(Field<IZeroForm, nTuple<THREE, TVL> , TL> const & lhs
		,Field<IZeroForm, nTuple<THREE, TVR> , TR> const & rhs)
{
	typedef BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TL>
			,Field<IZeroForm, nTuple<THREE, TVR> , TR> ,vector_calculus::OpDot> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<nTuple<N, TVL, TLExpr> , Field<IZeroForm, nTuple<N, TVR> , TRExpr>
		,vector_calculus::OpDot>
{

	typedef nTuple<N, TVL, TLExpr> TL;
	typedef Field<IZeroForm, nTuple<N, TVR> , TRExpr> TR;
	typedef TL LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, vector_calculus::OpDot> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<N, TVL, TLExpr> , nTuple<N, TVR> ,
	vector_calculus::OpDot>::ValueType ValueType;
	typedef typename TR::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (rhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Dot(lhs, rhs[s]));
	}

};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
typename BiOp<nTuple<N, TVL, TLExpr> , Field<IZeroForm, nTuple<N, TVR> , TRExpr>
		,vector_calculus::OpDot>::ResultType //
Dot(nTuple<N, TVL, TLExpr> const & lhs
		, Field<IZeroForm, nTuple<N, TVR> , TRExpr> const &rhs)
{
	typedef typename BiOp<nTuple<N, TVL, TLExpr>
			, Field<IZeroForm, nTuple<N, TVR> , TRExpr> ,vector_calculus::OpDot>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, nTuple<N, TVL> , TLExpr> ,nTuple<N, TVR, TRExpr>
		,vector_calculus::OpDot>
{

	typedef Field<IZeroForm, nTuple<N, TVL> , TLExpr> TL;
	typedef nTuple<N, TVR, TRExpr> TR;

	typedef typename TL::ConstReference LReference;
	typedef TR RReference;

	typedef BiOp<TL, TR, vector_calculus::OpDot> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<N, TVL> , nTuple<N, TVR, TRExpr>
			, vector_calculus::OpDot>::ValueType ValueType;
	typedef typename TL::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Dot(lhs[s], rhs));
	}

};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline typename BiOp<Field<IZeroForm, nTuple<N, TVL> , TLExpr>
		,nTuple<N, TVR, TRExpr> ,vector_calculus::OpDot>::ResultType //
Dot(Field<IZeroForm, nTuple<N, TVL> , TLExpr> const & lhs
		,nTuple<N, TVR, TRExpr> const & rhs)
{
	typedef typename BiOp<Field<IZeroForm, nTuple<N, TVL> , TLExpr>
			,nTuple<N, TVR, TRExpr> ,vector_calculus::OpDot>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TLExpr>
		,Field<IZeroForm, nTuple<THREE, TVR> , TRExpr> ,vector_calculus::OpCross>
{

	typedef Field<IZeroForm, nTuple<THREE, TVL> , TLExpr> TL;
	typedef Field<IZeroForm, nTuple<THREE, TVR> , TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, vector_calculus::OpCross> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<THREE, TVL> , nTuple<THREE, TVR>
			, vector_calculus::OpCross>::ValueType ValueType;
	typedef typename TL::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Cross(lhs[s], rhs[s]));
	}
};

template<typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TL>
		,Field<IZeroForm, nTuple<THREE, TVR> , TR> ,vector_calculus::OpCross>::ResultType //
Cross(Field<IZeroForm, nTuple<THREE, TVL> , TL> const & lhs
		,Field<IZeroForm, nTuple<THREE, TVR> , TR> const & rhs)
{
	typedef typename BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TL>
			,Field<IZeroForm, nTuple<THREE, TVR> , TR> ,vector_calculus::OpCross>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<nTuple<N, TVL, TLExpr> , Field<IZeroForm, nTuple<N, TVR> , TRExpr>
		,vector_calculus::OpCross>
{

	typedef nTuple<N, TVL, TLExpr> TL;
	typedef Field<IZeroForm, nTuple<N, TVR> , TRExpr> TR;
	typedef TL LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, vector_calculus::OpCross> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<N, TVL, TLExpr> , nTuple<N, TVR>
			, vector_calculus::OpCross>::ValueType ValueType;
	typedef typename TR::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (rhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Cross(lhs, rhs[s]));
	}

};

template<int N, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
typename BiOp<nTuple<N, TVL, TLExpr> , Field<IZeroForm, nTuple<N, TVR> , TRExpr>
		,vector_calculus::OpCross>::ResultType //
Cross(nTuple<N, TVL, TLExpr> const & lhs
		, Field<IZeroForm, nTuple<N, TVR> , TRExpr> const &rhs)
{
	typedef typename BiOp<nTuple<N, TVL, TLExpr>
			, Field<IZeroForm, nTuple<N, TVR> , TRExpr>
			,vector_calculus::OpCross>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TLExpr>
		,nTuple<THREE, TVR, TRExpr> ,vector_calculus::OpCross>
{

	typedef Field<IZeroForm, nTuple<THREE, TVL> , TLExpr> TL;
	typedef nTuple<THREE, TVR, TRExpr> TR;

	typedef typename TL::ConstReference LReference;
	typedef TR RReference;

	typedef BiOp<TL, TR, vector_calculus::OpCross> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<nTuple<THREE, TVL>
			, nTuple<THREE, TVR, TRExpr> , vector_calculus::OpCross>::ValueType ValueType;
	typedef typename TL::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (Cross(lhs[s], rhs));
	}

};

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
inline typename BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TLExpr>
		,nTuple<THREE, TVR, TRExpr> ,vector_calculus::OpCross>::ResultType //
Cross(Field<IZeroForm, nTuple<THREE, TVL> , TLExpr> const & lhs
		, nTuple<THREE, TVR, TRExpr> const & rhs)
{
	typedef typename BiOp<Field<IZeroForm, nTuple<THREE, TVL> , TLExpr>
			, nTuple<THREE, TVR, TRExpr> ,vector_calculus::OpCross>::ResultType ResultType;
	return (ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr>
struct UniOp<Field<IZeroForm, TVL, TLExpr> ,vector_calculus::OpGrad>
{

	typedef Field<IZeroForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<IZeroForm, TVL, TLExpr> , vector_calculus::OpGrad> ThisType;
	static const int IForm = IOneForm;
	typedef TVL ValueType;
	typedef typename TL::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.grad_(lhs, s));
	}
};

template<typename TVL, typename TL>
inline typename UniOp<Field<IZeroForm, TVL, TL> , vector_calculus::OpGrad>::ResultType //
Grad(Field<IZeroForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<IZeroForm, TVL, TL> , vector_calculus::OpGrad>::ResultType ResultType;
	return (ResultType(lhs));
}

template<typename TVL, typename TLExpr>
struct UniOp<Field<IOneForm, TVL, TLExpr> ,vector_calculus::OpDiverge>
{

	typedef Field<IOneForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<IOneForm, TVL, TLExpr> , vector_calculus::OpDiverge> ThisType;
	static const int IForm = IZeroForm;
	typedef TVL ValueType;
	typedef typename TL::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}

	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.diverge_(lhs, s));
	}
};

template<typename TVL, typename TL>
inline typename UniOp<Field<IOneForm, TVL, TL> , vector_calculus::OpDiverge>::ResultType //
Diverge(Field<IOneForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<IOneForm, TVL, TL> , vector_calculus::OpDiverge>::ResultType ResultType;
	return (ResultType(lhs));
}

template<typename TVL, typename TLExpr>
struct UniOp<Field<IOneForm, TVL, TLExpr> ,vector_calculus::OpCurl>
{
	typedef Field<IOneForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<IOneForm, TVL, TLExpr> , vector_calculus::OpCurl> ThisType;
	static const int IForm = ITwoForm;
	typedef TVL ValueType;
	typedef typename Field<IOneForm, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.curl_(lhs, s));
	}
};

template<typename TVL, typename TL>
inline typename UniOp<Field<IOneForm, TVL, TL> , vector_calculus::OpCurl>::ResultType //
Curl(Field<IOneForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<IOneForm, TVL, TL> , vector_calculus::OpCurl>::ResultType ResultType;
	return (ResultType(lhs));
}

template<typename TVL, typename TLExpr>
struct UniOp<Field<ITwoForm, TVL, TLExpr> ,vector_calculus::OpCurl>
{
	typedef Field<ITwoForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<ITwoForm, TVL, TLExpr> , vector_calculus::OpCurl> ThisType;
	static const int IForm = IOneForm;
	typedef TVL ValueType;
	typedef typename Field<ITwoForm, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.curl_(lhs, s));
	}
};

template<typename TVL, typename TL>
inline typename UniOp<Field<ITwoForm, TVL, TL> , vector_calculus::OpCurl>::ResultType //
Curl(Field<ITwoForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<ITwoForm, TVL, TL> , vector_calculus::OpCurl>::ResultType ResultType;
	return (ResultType(lhs));
}

template<int IPD, typename TVL, typename TLExpr>
struct UniOp<Field<IOneForm, TVL, TLExpr> ,vector_calculus::OpCurlPD<IPD> >
{
	typedef Field<IOneForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<TL, vector_calculus::OpCurlPD<IPD> > ThisType;
	static const int IForm = ITwoForm;
	typedef TVL ValueType;
	typedef typename Field<IOneForm, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.template curlPd_<IPD>(lhs, s));
	}
};
template<int IPD, typename TVL, typename TL>
inline typename UniOp<Field<IOneForm, TVL, TL> , vector_calculus::OpCurlPD<IPD> >::ResultType //
CurlPD(Field<IOneForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<IOneForm, TVL, TL>
			, vector_calculus::OpCurlPD<IPD> >::ResultType ResultType;
	return (ResultType(lhs));
}

template<int IPD, typename TVL, typename TLExpr>
struct UniOp<Field<ITwoForm, TVL, TLExpr> ,vector_calculus::OpCurlPD<IPD> >
{

	typedef Field<ITwoForm, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<ITwoForm, TVL, TLExpr> , vector_calculus::OpCurlPD<IPD> > ThisType;
	static const int IForm = IOneForm;
	typedef TVL ValueType;
	typedef typename Field<ITwoForm, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (lhs.grid.template curlPd_<IPD>(lhs, s));
	}
};

template<int IPD, typename TVL, typename TL>
inline typename UniOp<Field<ITwoForm, TVL, TL> , vector_calculus::OpCurlPD<IPD> >::ResultType //
CurlPD(Field<ITwoForm, TVL, TL> const & lhs)
{
	typedef typename UniOp<Field<ITwoForm, TVL, TL>
			, vector_calculus::OpCurlPD<IPD> >::ResultType ResultType;
	return (ResultType(lhs));
}

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
