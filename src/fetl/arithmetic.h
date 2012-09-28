/*
 * arithmetic.h
 *
 *  Created on: 2012-3-1
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef ARITHMETIC_H_
#define ARITHMETIC_H_
#include "fetl/fetl_defs.h"
#include "primitives/operation.h"
namespace simpla
{
namespace fetl
{

// Arithmetic
//-----------------------------------------

template<int IFORM, typename TVL, typename TLExpr>
struct UniOp<Field<IFORM, TVL, TLExpr> , arithmetic::OpNegative>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef TL LReference;
	typedef UniOp<Field<IFORM, TVL, TLExpr> , arithmetic::OpNegative> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, NullType, arithmetic::OpNegative>::ValueType ValueType;
	typedef typename Field<IFORM, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, size_t const & s)
	{
		return (-lhs[s]);
	}

};

template<int IFORM, typename TV, typename TL> //
inline typename UniOp<Field<IFORM, TV, TL> ,arithmetic::OpNegative>::ResultType //
operator -(Field<IFORM, TV, TL> const & lhs)
{
	typedef UniOp<Field<IFORM, TV, TL> ,arithmetic::OpNegative> TOP;
	return (typename TOP::ResultType(lhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IFORM, TVL, TLExpr> ,Field<IFORM, TVR, TRExpr>
		,arithmetic::OpAddition>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef Field<IFORM, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpAddition> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpAddition>::ValueType ValueType;
	typedef typename Field<IFORM, TVL, TLExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] + rhs[s]);
	}

};
template<int IFORM, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IFORM, TVL, TL> ,Field<IFORM, TVR, TR>
		,arithmetic::OpAddition>::ResultType //
operator +(Field<IFORM, TVL, TL> const &lhs , Field<IFORM, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IFORM, TVL, TL> ,Field<IFORM, TVR, TR>
			,arithmetic::OpAddition> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IFORM, TVL, TLExpr> ,Field<IFORM, TVR, TRExpr>
		,arithmetic::OpSubtraction>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef Field<IFORM, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpSubtraction> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpSubtraction>::ValueType ValueType;
	typedef typename Field<IFORM, TVL, TLExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] - rhs[s]);
	}

};
template<int IFORM, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IFORM, TVL, TL> ,Field<IFORM, TVR, TR>
		,arithmetic::OpSubtraction>::ResultType //
operator -(Field<IFORM, TVL, TL> const &lhs , Field<IFORM, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IFORM, TVL, TL> ,Field<IFORM, TVR, TR>
			,arithmetic::OpSubtraction> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<typename TVL, int IFORM, typename TVR, typename TRExpr>
struct BiOp<TVL, Field<IFORM, TVR, TRExpr> ,arithmetic::OpMultiplication>
{

	typedef TVL TL;
	typedef Field<IFORM, TVR, TRExpr> TR;
	typedef TVL LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpMultiplication> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR,
			typename arithmetic::OpMultiplication>::ValueType ValueType;
	typedef typename Field<IFORM, TVR, TRExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (rhs.grid);
	}
	static ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs * rhs[s]);
	}

};
template<typename TL, int IFORM, typename TV, typename TR>
inline typename BiOp<TL, Field<IFORM, TV, TR> ,arithmetic::OpMultiplication>::ResultType //
operator *(TL const & lhs, Field<IFORM, TV, TR> const &rhs)
{
	typedef BiOp<TL, Field<IFORM, TV, TR> ,arithmetic::OpMultiplication> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR>
struct BiOp<Field<IFORM, TVL, TLExpr> ,TVR,arithmetic::OpMultiplication>
{

	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef TVR TR;

	typedef typename TL::ConstReference LReference;
	typedef TVR RReference;

	typedef BiOp<TL, TR, arithmetic::OpMultiplication> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef typename Field<IFORM, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] * rhs);
	}

};
template<int IFORM, typename TV, typename TL, typename TR>
inline typename BiOp<Field<IFORM, TV, TL> ,TR ,arithmetic::OpMultiplication>::ResultType //
operator *(Field<IFORM, TV, TL> const &lhs, TR const & rhs)
{
	typedef BiOp<Field<IFORM, TV, TL> ,TR ,arithmetic::OpMultiplication> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr, int IFORM, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, TVL, TLExpr> ,Field<IFORM, TVR, TRExpr>
		,arithmetic::OpMultiplication>
{

	typedef Field<IZeroForm, TVL, TLExpr> TL;
	typedef Field<IFORM, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpMultiplication> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef typename Field<IFORM, TVR, TRExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs.grid.mapto_(Int2Type<IForm>(), lhs, s) * rhs[s]);
	}

};

template<int IFORM, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IZeroForm, TVL, TL> ,Field<IFORM, TVR, TR>
		,arithmetic::OpMultiplication>::ResultType //
operator *(Field<IZeroForm, TVL, TL> const &lhs
		, Field<IFORM, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IZeroForm, TVL, TL> ,Field<IFORM, TVR, TR>
			,arithmetic::OpMultiplication> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IFORM, TVL, TLExpr> ,Field<IZeroForm, TVR, TRExpr>
		,arithmetic::OpMultiplication>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef Field<IZeroForm, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpMultiplication> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef typename Field<IFORM, TVR, TRExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] * rhs.grid.mapto_(Int2Type<IForm>(), rhs, s));
	}
};

template<int IFORM, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IFORM, TVL, TL> ,Field<IZeroForm, TVR, TR>
		,arithmetic::OpMultiplication>::ResultType //
operator *(Field<IFORM, TVL, TL> const &lhs
		, Field<IZeroForm, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IFORM, TVL, TL> ,Field<IZeroForm, TVR, TR>
			,arithmetic::OpMultiplication> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IZeroForm, TVL, TLExpr> ,Field<IZeroForm, TVR, TRExpr>
		,arithmetic::OpMultiplication>
{
	typedef Field<IZeroForm, TVL, TLExpr> TL;
	typedef Field<IZeroForm, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpMultiplication> ThisType;
	static const int IForm = IZeroForm;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	typedef typename Field<IZeroForm, TVR, TRExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] * rhs[s]);
	}
};

template<typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IZeroForm, TVL, TL> ,Field<IZeroForm, TVR, TR>
		,arithmetic::OpMultiplication>::ResultType //
operator *(Field<IZeroForm, TVL, TL> const &lhs
		, Field<IZeroForm, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IZeroForm, TVL, TL> ,Field<IZeroForm, TVR, TR>
			,arithmetic::OpMultiplication> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR>
struct BiOp<Field<IFORM, TVL, TLExpr> ,TVR,arithmetic::OpDivision>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef TVR TR;

	typedef typename TL::ConstReference LReference;
	typedef TVR RReference;

	typedef BiOp<TL, TR, arithmetic::OpDivision> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, typename arithmetic::OpDivision>::ValueType ValueType;
	typedef typename Field<IFORM, TVL, TLExpr>::Grid Grid;
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] / rhs);
	}
};

template<int IFORM, typename TV, typename TL, typename TR>
inline typename BiOp<Field<IFORM, TV, TL> ,TR ,arithmetic::OpDivision>::ResultType //
operator /(Field<IFORM, TV, TL> const &lhs, TR const & rhs)
{
	typedef BiOp<Field<IFORM, TV, TL> ,TR ,arithmetic::OpDivision> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
struct BiOp<Field<IFORM, TVL, TLExpr> ,Field<IZeroForm, TVR, TRExpr>
		,arithmetic::OpDivision>
{
	typedef Field<IFORM, TVL, TLExpr> TL;
	typedef Field<IZeroForm, TVR, TRExpr> TR;
	typedef typename TL::ConstReference LReference;
	typedef typename TR::ConstReference RReference;

	typedef BiOp<TL, TR, arithmetic::OpDivision> ThisType;
	static const int IForm = IFORM;
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpDivision>::ValueType ValueType;
	typedef typename Field<IFORM, TVR, TRExpr>::Grid Grid; // FIXME need grid detriment
	typedef Field<IForm, ValueType, ThisType> ResultType;

	static Grid const & get_grid(TL const & lhs, TR const & rhs)
	{
		return (lhs.grid);
	}
	static inline ValueType op(TL const & lhs, TR const & rhs, size_t s)
	{
		return (lhs[s] / rhs.grid.mapto_(Int2Type<IForm>(), rhs, s));
	}
};

template<int IFORM, typename TVL, typename TL, typename TVR, typename TR>
inline typename BiOp<Field<IFORM, TVL, TL> ,Field<IZeroForm, TVR, TR>
		,arithmetic::OpDivision>::ResultType //
operator /(Field<IFORM, TVL, TL> const &lhs
		, Field<IZeroForm, TVR, TR> const & rhs)
{
	typedef BiOp<Field<IFORM, TVL, TL> ,Field<IZeroForm, TVR, TR>
			,arithmetic::OpDivision> TOP;
	return (typename TOP::ResultType(lhs, rhs));
}

template<int IFORM, typename TVL, typename TLExpr, typename TVR, typename TRExpr>
typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType //
InnerProduct(Field<IFORM, TVL, TLExpr> const & lhs
		, Field<IFORM, TVR, TRExpr> const & rhs)
{
	typedef typename TypeOpTraits<TVL, TVR, arithmetic::OpMultiplication>::ValueType ValueType;
	ValueType res;
	res = 0;
	return (lhs.grid.InnerProduct(lhs, rhs));

}

} // namespace fetl
}
// namespace simpla

#endif /* ARITHMETIC_H_ */
