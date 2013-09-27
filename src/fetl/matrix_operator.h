/*
 * matrix_operator.h
 *
 *  Created on: 2013-7-10
 *      Author: salmon
 */

#ifndef MATRIX_OPERATOR_H_
#define MATRIX_OPERATOR_H_

#include "fetl_defs.h"
namespace simpla
{

template<typename T>
Real MaxNorm(T const & m)
{
	return (m);
}

template<typename T>
Real MaxNorm(std::complex<T> const & m)
{

	return (abs(m));
}

template<int N, typename T>
Real MaxNorm(nTuple<N, T> const & m)
{
	Real res = 0.0;

	for (size_t s = 0; s < N; ++s)
	{
		res = (res > MaxNorm(m[s])) ? res : MaxNorm(m[s]);
	}

	return (res);
}

template<typename TG, int IFORM, typename TF>
Real MaxNorm(Field<TG, IFORM, TF> const & m)
{
	TG const& grid = m.grid;
	size_t num_of_eles = grid.num_of_elements(IFORM);

	Real res = 0.0;

	for (size_t s = 0; s < num_of_eles; ++s)
	{
		res = (res > MaxNorm(m[s])) ? res : MaxNorm(m[s]);
	}

	return (res);

}

template<typename T>
Real InnerProduct(T const & l, T const & r)
{

	return (l * r);
}

template<int N, typename T>
Real InnerProduct(nTuple<N, T> const & l, nTuple<N, T> const & r)
{
	Real res = 0.0;

	for (size_t s = 0; s < N; ++s)
	{
		res += InnerProduct(l[s], r[s]);
	}

	return (res);
}

template<typename TG, int IFORM, typename TF, typename TR>
Real InnerProduct(Field<TG, IFORM, TL> const & l,
		Field<TG, IFORM, TR> const & r)
{
	TG const& grid = l.grid;
	size_t num_of_eles = grid.num_of_elements(IFORM);

	Real res = 0.0;

	for (size_t s = 0; s < num_of_eles; ++s)
	{
		res += InnerProduct(l[s], r[s]);
	}

	return (res);

}

}  // namespace simpla

#endif /* MATRIX_OPERATOR_H_ */
