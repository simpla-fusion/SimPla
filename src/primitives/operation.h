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

namespace arithmetic
{

struct OpMultiplication;
struct OpDivision;
struct OpAddition;
struct OpSubtraction;
struct OpNegative;

} // namespace arithmetic
namespace vector_calculus
{

struct OpDot;
struct OpCross;

struct OpGrad;
struct OpDiverge;
struct OpCurl;
template<int IR> struct OpCurlPD;

} // namespace vector_calculus
template<int N, typename T, typename TExpr> struct nTuple;
template<typename, typename, typename > struct BiOp;
template<typename, typename > struct UniOp;

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

template<typename, typename, typename TOP> struct BiOp;
template<typename, typename TOP> struct UniOp;

template<typename, typename, typename TOP> struct TypeOpTraits;

// NOTE: this is a replacement before C++11 available for CUDA
template<typename TVL>
struct TypeOpTraits<TVL, NullType, arithmetic::OpNegative>
{
	typedef TVL ValueType;
};

template<typename TOP>
struct TypeOpTraits<int, int, TOP>
{
	typedef int ValueType;
};
template<typename TOP>
struct TypeOpTraits<long, long, TOP>
{
	typedef long ValueType;
};
template<typename TOP>
struct TypeOpTraits<unsigned long, unsigned long, TOP>
{
	typedef unsigned long ValueType;
};
template<typename TOP>
struct TypeOpTraits<double, double, TOP>
{
	typedef double ValueType;
};
template<typename TOP>
struct TypeOpTraits<std::complex<double>, std::complex<double>, TOP>
{
	typedef std::complex<double> ValueType;
};
#define TYPE_OP_RULE(TL,TR,TV) \
template<typename TOP> struct TypeOpTraits<TL, TR, TOP> { 	typedef TV ValueType; }; \
template<typename TOP> struct TypeOpTraits<TR, TL, TOP> { 	typedef TV ValueType; };

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

}// namespace simpla

#endif /* OPERATION_H_ */
