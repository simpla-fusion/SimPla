/*
 * utilities.h
 *
 *  Created on: 2013年11月24日
 *      Author: salmon
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <sstream>
#include <string>

namespace simpla
{

template<typename T>
inline std::string ToString(T const & v)
{
	std::ostringstream os;
	os << v;
	return os.str();
}
template<typename T>
inline T ToValue(std::string const & str)
{
	T v;
	std::istringstream os(str);
	os >> v;
	return std::move(v);
}

template<int N, typename T>
inline std::string ToString(nTuple<N, T> const & v, std::string const & sep =
		" ")
{

	std::ostringstream os;
	for (int i = 0; i < N - 1; ++i)
	{
		os << v[i] << sep;
	}
	os << v[N - 1];
	return (os.str());
}

}  // namespace simpla

#endif /* UTILITIES_H_ */
