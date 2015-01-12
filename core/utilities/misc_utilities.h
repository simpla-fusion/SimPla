/**
 * @file misc_utilities.h
 *
 *  created on: 2013-11-24
 *      Author: salmon
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "../gtl/ntuple.h"
/**
 *  @ingroup    utilities
 */
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

template<unsigned int N, typename T>
inline std::string ToString(nTuple<T, N> const & v, std::string const & sep =
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

inline std::string AutoIncrease(std::function<bool(std::string)> const & fun,
		size_t count = 0, unsigned int width = 4)
{
	std::string res("");
	while (fun(res))
	{
		std::ostringstream os;

		os << std::setw(width) << std::setfill('0') << count;
		++count;
		res = os.str();
	}
	return res;
}

inline bool CheckFileExists(std::string const & name)
{
	if (FILE *file = fopen(name.c_str(), "r"))
	{
		fclose(file);
		return true;
	}
	else
	{
		return false;
	}
}

template<typename T>
inline unsigned long make_hash(T s)
{
	return static_cast<unsigned long>(s);
}
template<typename TR> size_t size_of_range(TR const &range)
{
	size_t count = 0;
	for (auto const & item : range)
	{
		++count;
	}
	return count;
}

template<typename T>
bool find_same(T const & left)
{
	return false;
}
template<typename T, typename ...Others>
bool find_same(T const & left, T const & first, Others && ... others)
{
	return left == first || equal_to(left, std::forward<Others>( others)...);
}
}
// namespace simpla

#endif /* UTILITIES_H_ */
