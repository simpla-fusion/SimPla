/*
 * covector.h
 *
 *  Created on: 2013年8月14日
 *      Author: salmon
 */

#ifndef COVECTOR_H_
#define COVECTOR_H_
#include <map>
namespace simpla
{
template<typename T>
struct CoVector: public std::map<size_t, T>
{

};
template<typename T>
CoVector<T> && operator+(CoVector<T> && lhs, CoVector<T> && rhs)
{
	for (auto v : rhs)
	{
		try
		{
			lhs.at(v.first) += v.second;
		} catch (...)
		{
			lhs.insert(v);
		}
	}
}

template<typename T>
CoVector<T> && operator-(CoVector<T> && lhs, CoVector<T> && rhs)
{
	for (auto v : rhs)
	{
		try
		{
			lhs.at(v.first) -= v.second;
		} catch (...)
		{
			lhs.insert(v);
		}
	}
}

template<typename T>
CoVector<T> && operator*(CoVector<T> && lhs, T const & rhs)
{

	for (auto v : lhs)
	{
		v.second *= rhs;
	}
	return (std::move(lhs));
}

template<typename T, typename TR>
CoVector<T> && operator*(CoVector<T> && lhs, TR const & rhs)
{
	CoVector<decltype(T()*rhs)> res;

	for (auto v : lhs)
	{
		res.insert(std::make_pair(v.first, v.second * rhs))
	}
	return (std::move(res));
}
}  // namespace simpla

#endif /* COVECTOR_H_ */
