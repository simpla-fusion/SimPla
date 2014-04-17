/*
 * matrix.h
 *
 *  Created on: 2013年8月14日
 *      Author: salmon
 */

#ifndef MATRIX_H_
#define MATRIX_H_
#include "expression.h"

namespace simpla
{
/**
 *  Field is Vector
 *  CoVecotr \times  Vector \mapsto\mathbb{R}
 *
 * */
template<typename TV>
class CoVector: public std::map<size_t, TV>
{
public:
	typedef TV ValueType;
};
template<template<typename > class TOP, typename TL>
struct CoVector<UniOp<TOP, TL> >
{
	typename ConstReferenceTraits<TL>::type l_;

	CoVector(TL const & l) :
			l_(l)
	{
	}
	inline auto operator[](size_t s) const
	DECL_RET_TYPE ((TOP<TL>::eval(l_, s)))

}	;

}
// namespace simpla

#endif /* MATRIX_H_ */
