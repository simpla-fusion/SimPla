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

}  // namespace simpla

#endif /* UTILITIES_H_ */
