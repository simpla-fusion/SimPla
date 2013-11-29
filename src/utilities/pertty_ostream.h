/*
 * pertty_ostream.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef PERTTY_OSTREAM_H_
#define PERTTY_OSTREAM_H_

#include <iostream>
#include <vector>

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
template<typename T, typename TV> std::basic_ostream<T>&
operator<<(std::basic_ostream<T>& out_s, std::vector<TV> const& a)
{
	out_s << "{";

	for (auto & v : a)
	{
		out_s << v << ",";
	}
	out_s << "\b}";
	return out_s;

}

}  // namespace simpla

#endif /* PERTTY_OSTREAM_H_ */
