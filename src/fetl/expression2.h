/*
 * expression2.h
 *
 *  Created on: 2013年8月2日
 *      Author: salmon
 */

#ifndef EXPRESSION2_H_
#define EXPRESSION2_H_

#include <cstddef>

namespace simpla
{
/**
 * scalar,complex,ntuple,ntuple<complex>,
 * field<T>
 *
 *
 * */
#define DECL_RET_TYPE(_EXPR_) ->decltype((_EXPR_)){return (_EXPR_);}
template<typename > struct Expression;

template<template<typename, typename > class TOP, typename TL, typename TR>
class Expression<TOP<TL, TR> >
{
public:

private:
	TL l_;
	TR r_;
};

template<typename > class OpNegate
{
	template<typename T>
	static inline auto eval(T const & r) DECL_RET_TYPE((-r))

	template<typename T>
	static inline auto eval(T const & r, size_t s) DECL_RET_TYPE((-r[s]))
};
template<typename > class OpUnaryPlus
{
	template<typename T>
	static inline auto eval(T const & r) DECL_RET_TYPE((r))

	template<typename T>
	static inline auto eval(T const & r, size_t s) DECL_RET_TYPE((r[s]))
};
template<typename, typename > class OpPlus
{
	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r)
	DECL_RET_TYPE((l+r))

	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r, size_t s)
	DECL_RET_TYPE((l[s]+r[s]))
};
template<typename, typename > class OpMinus
{
	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r)
	DECL_RET_TYPE((l-r))

	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r, size_t s)
	DECL_RET_TYPE((l[s]-r[s]))
};
template<typename, typename > class OpMultiplies
{
	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r)
	DECL_RET_TYPE((l*r))

	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r, size_t s)
	DECL_RET_TYPE((l[s]*r[s]))
};

template<typename, typename > class OpDivides
{
	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r)
	DECL_RET_TYPE((l/r))

	template<typename TL, typename TR>
	static inline auto eval(TL const & l, TR const & r, size_t s)
	DECL_RET_TYPE((l[s]/r[s]))
};
}
// namespace simpla

#endif /* EXPRESSION2_H_ */
