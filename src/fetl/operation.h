/*
 * operation.h
 *
 *  Created on: 2012-3-23
 *      Author: salmon
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 * STUPID!!   DO NOT CHANGE THIS EXPRESSION TEMPLATES WITHOUT REALLY REALLY GOOD REASON!!!!!
 *
 *!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */

#ifndef OPERATION_H_
#define OPERATION_H_
namespace simpla
{
namespace fetl
{
template<typename T> struct TypeTraits
{
	typedef T Reference;
	typedef const T ConstReference;
};

template<int N, typename T, typename TExpr> struct nTuple;
template<int IFORM, typename TV, typename > struct Field;

template<typename, typename, template<typename, typename > class TOP> struct TypeOpTraits;

template<template<typename, typename > class TOP>
struct TypeOpTraits<int, int, TOP>
{
	typedef int ValueType;
};

template<template<typename, typename > class TOP>
struct TypeOpTraits<long, long, TOP>
{
	typedef long ValueType;
};

template<template<typename, typename > class TOP>
struct TypeOpTraits<double, double, TOP>
{
	typedef double ValueType;
};

template<template<typename, typename > class TOP>
struct TypeOpTraits<std::complex<double>, std::complex<double>, TOP>
{
	typedef std::complex<double> ValueType;
};


template<int IFORM, typename TL, typename TEXPL, typename TR, template<typename,
		typename > class TOP>
struct TypeOpTraits<Field<IFORM, TL, TEXPL>, TR, TOP>
{
	typedef typename TypeOpTraits<TL, TR, TOP>::ValueType ValueType;
};
template<typename TL, int IFORM, typename TR, typename TEXPR, template<typename,
		typename > class TOP>
struct TypeOpTraits<TL, Field<IFORM, TR, TEXPR>, TOP>
{
	typedef typename TypeOpTraits<TL, TR, TOP>::ValueType ValueType;
};

template<int IFORML, typename TLV, typename TEXPL, int IFORMR, typename TRV,
		typename TEXPR, template<typename, typename > class TOP>
struct TypeOpTraits<Field<IFORML, TLV, TEXPL>, Field<IFORMR, TRV, TEXPR>, TOP>
{
	typedef typename TypeOpTraits<TLV, TRV, TOP>::ValueType ValueType;
};

#define TYPE_OP_RULE(TL,TR,TV) \
template<template<typename, typename > class TOP> struct TypeOpTraits<TL, TR, TOP> { 	typedef TV ValueType; }; \
template<template<typename, typename > class TOP> struct TypeOpTraits<TR, TL, TOP> { 	typedef TV ValueType; };

TYPE_OP_RULE(double, int, double)
TYPE_OP_RULE(double, long, double)
TYPE_OP_RULE(double, unsigned int, double)
TYPE_OP_RULE(double, unsigned long, double)
TYPE_OP_RULE(std::complex<double>, double, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, int, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, unsigned int, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, long, std::complex<double>)
TYPE_OP_RULE(std::complex<double>, unsigned long, std::complex<double>)
#undef TYPE_OP_RULE

//template<typename TL, typename TR>
//struct TypeOpTraits<TL, TR, arithmetic::OpMultiplication>
//{
//	typedef decltype(TL() * TR()) ValueType;
//};
//template<typename TL, typename TR>
//struct TypeOpTraits<TL, TR, arithmetic::OpDivision>
//{
//	typedef decltype(TL() / TR()) ValueType;
//};
//template<typename TL, typename TR>
//struct TypeOpTraits<TL, TR, arithmetic::OpAddition>
//{
//	typedef decltype(TL() + TR()) ValueType;
//};
//template<typename TL, typename TR>
//struct TypeOpTraits<TL, TR, arithmetic::OpSubtraction>
//{
//	typedef decltype(TL() - TR()) ValueType;
//};
////---------------------------------------------------------------------------------------------

////---------------------------------------------------------------------------------------------

template<typename T> struct ComplexTraits
{
	typedef std::complex<T> ValueType;
};

template<typename T> struct ComplexTraits<std::complex<T> >
{
	typedef std::complex<T> ValueType;
};

template<int N, typename T> struct ComplexTraits<nTuple<N, T, NullType> >
{
	typedef nTuple<N, typename ComplexTraits<T>::ValueType, NullType> ValueType;
};

////---------------------------------------------------------------------------------------------

namespace arithmetic
{
template<typename TL, typename TR>
struct OpMultiplication
{
	typedef typename TypeOpTraits<TL, TR, OpMultiplication>::ValueType ValueType;

	static inline ValueType eval(TL const &l, TR const & r)
	{
		return (l * r);
	}
}
;
template<typename TL, typename TR>
struct OpDivision
{
	typedef typename TypeOpTraits<TL, TR, OpDivision>::ValueType ValueType;

	static inline ValueType eval(TL const &l, TR const & r)
	{
		return (l / r);
	}
}
;
template<typename TL, typename TR>
struct OpAddition
{
	typedef typename TypeOpTraits<TL, TR, OpAddition>::ValueType ValueType;

	static inline ValueType eval(TL const &l, TR const & r)
	{
		return (l + r);
	}
}
;
template<typename TL, typename TR>
struct OpSubtraction
{
	typedef typename TypeOpTraits<TL, TR, OpSubtraction>::ValueType ValueType;

	static inline ValueType eval(TL const &l, TR const & r)
	{
		return (l - r);
	}
}
;
template<typename TL>
struct OpNegative
{
	typedef TL ValueType;

	static inline TL eval(TL const &l)
	{
		return (-l);
	}
}
;

} // namespace arithmetic
namespace vector_calculus
{

template<typename TL, typename TR> struct OpDot;
template<typename TL, typename TR> struct OpCross;

template<typename TL> struct OpGrad;
template<typename TL> struct OpDiverge;
template<typename TL> struct OpCurl;
template<int IR, typename TL> struct OpCurlPD;

} // namespace vector_calculus

// NOTE: this is a replacement before C++11 available for CUDA

//template<template<typename, typename > class TOP>
//struct TypeOpTraits<int, int, TOP>
//{
//	typedef int ValueType;
//};
//template<template<typename, typename > class TOP>
//struct TypeOpTraits<long, long, TOP>
//{
//	typedef long ValueType;
//};
//template<template<typename, typename > class TOP>
//struct TypeOpTraits<unsigned long, unsigned long, TOP>
//{
//	typedef unsigned long ValueType;
//};
//template<template<typename, typename > class TOP>
//struct TypeOpTraits<double, double, TOP>
//{
//	typedef double ValueType;
//};
//template<template<typename, typename > class TOP>
//struct TypeOpTraits<std::complex<double>, std::complex<double>, TOP>
//{
//	typedef std::complex<double> ValueType;
//};

}//namespace fetl
} // namespace simpla

#endif /* OPERATION_H_ */
