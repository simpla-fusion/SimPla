/*
 * pertty_ostream.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef PERTTY_OSTREAM_H_
#define PERTTY_OSTREAM_H_

#include <istream>
#include <ostream>
#include <vector>
#include <map>

namespace simpla
{

template<typename T, typename TV> std::basic_istream<T>&
operator>>(std::basic_istream<T>& in_s, std::vector<TV> & a)
{

	for (auto & v : a)
	{
		in_s >> v;
	}
	return in_s;

}

template<typename T, typename TX, typename TY> std::basic_istream<T>&
get(std::basic_istream<T>& in_s, size_t num, std::map<TX, TY> & a)
{

	for (size_t s = 0; s < num; ++s)
	{
		TX x;
		TY y;
		in_s >> x >> y;
		a.emplace(std::make_pair(x, y));

	}
	return in_s;

}
template<typename T, typename TV> std::basic_ostream<T>&
operator<<(std::basic_ostream<T>& out_s, std::vector<TV> const& a)
{

	for (auto & v : a)
	{
		out_s << v << " ";
	}
	return out_s;

}
template<typename T, typename TX, typename TY> std::basic_ostream<T>&
operator<<(std::basic_ostream<T>& out_s, std::map<TX, TY> const& a)
{

	for (auto const & v : a)
	{
		out_s << " " << v.first << " " << v.second << " ";
	}
	return out_s;

}
}  // namespace simpla

#endif /* PERTTY_OSTREAM_H_ */
