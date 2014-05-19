/*
 * sp_math.h
 *
 *  Created on: 2014年5月19日
 *      Author: salmon
 */

#ifndef SP_MATH_H_
#define SP_MATH_H_

namespace simpla
{

template<typename T>
inline T Mod(T const & v, T const & n)
{
	return v % n;
}

}  // namespace simpla

#endif /* SP_MATH_H_ */
