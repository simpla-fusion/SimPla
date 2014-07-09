/*
 * sp_math.h
 *
 *  created on: 2014-5-19
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
