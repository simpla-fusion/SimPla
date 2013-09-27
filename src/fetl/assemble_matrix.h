/*
 * assemble_matrix.h
 *
 *  Created on: 2013年8月14日
 *      Author: salmon
 */

#ifndef ASSEMBLE_MATRIX_H_
#define ASSEMBLE_MATRIX_H_

#include <type_traits>
#include <map>
namespace simpla
{

template<typename TA, typename TB, typename TE>
auto AssamebleMatrix(std::map<size_t, std::map<size_t, TA> > & m,
		std::map<size_t, TB> & b, TE & expr)
		->std::enable_if<is_LinearFunction<
		typename std::remove_reference<decltype(expr[0])>::type>::value,void>
{

	for (size_t s = 0, send = expr.get_num_of_element(); s < send; ++s)
	{
		b[s] = 0;
		expr[s].get_coeffs(m[s], b[s]);
	}
}

}
// namespace simpla

#endif /* ASSEMBLE_MATRIX_H_ */
