/*
 * utilities.h
 *
 *  Created on: 2013年11月24日
 *      Author: salmon
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>

#include "../fetl/ntuple.h"
#include "log.h"

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
inline std::string ToString(nTuple<N, T> const & v, std::string const & sep = " ")
{

	std::ostringstream os;
	for (int i = 0; i < N - 1; ++i)
	{
		os << v[i] << sep;
	}
	os << v[N - 1];
	return (os.str());
}

inline std::string AutoIncrease(std::function<bool(std::string)> const & fun, size_t count = 0, int width = 4)
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
inline void TheStart(int flag = 1)
{
	switch (flag)
	{
	default:
		LOGGER << "MISSION START!";
		INFORM << SINGLELINE;
		VERBOSE << "So far so good, let's start work! ";
		INFORM << "[MISSOIN     START]: " << TimeStamp;
		INFORM << SINGLELINE;
	}
}
inline void TheEnd(int flag = 1)
{
	switch (flag)
	{
	case -2:
		INFORM << "Oop! Some thing wrong! Don't worry, maybe not your fault!\n"
				" Just maybe! Please Check your configure file again! ";
		break;
	case -1:
		INFORM << "Sorry! I can't help you now! Please, Try again later!";
		break;
	case 0:
		INFORM << "See you! ";
		break;
	case 1:
	default:
		LOGGER << "MISSION COMPLETED!";

		INFORM << SINGLELINE;
		INFORM << "[MISSION COMPLETED]: " << TimeStamp;
		VERBOSE << "Job is Done!! ";
		VERBOSE << "	I'm so GOOD!";
		VERBOSE << "		Thanks me please!";
		VERBOSE << "			Thanks me please!";
		VERBOSE << "You are welcome!";
		INFORM << SINGLELINE;

	}
	LOGGER <<std::endl;
	INFORM<<std::endl;

	exit(1);
}

template<typename T>
inline unsigned long make_hash(T s)
{
	return static_cast<unsigned long>(s);
}
}  // namespace simpla

#endif /* UTILITIES_H_ */
