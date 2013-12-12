/*
 * pertty_stream.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef PERTTY_STREAM_H_
#define PERTTY_STREAM_H_

#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace simpla
{

template<typename TV, typename ...Others> std::istream&
operator>>(std::istream& is, std::vector<TV, Others...> & a)
{

	std::copy(std::istream_iterator<TV>(is), std::istream_iterator<TV>(),
			std::back_inserter(a));
	return is;

}

template<typename TX, typename TY, typename ...Others> std::istream&
get(std::istream& is, size_t num, std::map<TX, TY, Others...> & a)
{

	for (size_t s = 0; s < num; ++s)
	{
		TX x;
		TY y;
		is >> x >> y;
		a.emplace(std::make_pair(x, y));
	}
	return is;
}

template<typename TI>
std::ostream & ContainerOutPut1(std::ostream & os, TI it, TI ie)
{

	os << *it;

	size_t s = 0;

	for (++it; it != ie; ++it)
	{
		os << " , " << *it;

		++s;
		if (s % 10 == 0)
			os << std::endl;
	}

	return os;
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::vector<U, Others...> const &d)
{
	return ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::list<U, Others...> const &d)
{
	return ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::set<U, Others...> const &d)
{
	return ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os,
		std::multiset<U, Others...> const &d)
{
	return ContainerOutPut1(os, d.begin(), d.end());
}

template<typename TI>
std::ostream & ContainerOutPut2(std::ostream & os, TI it, TI ie)
{
	os << it->first << "=" << it->second;

	for (++it; it != ie; ++it)
	{
		os << " , " << it->first << " = " << it->second;
	}
	return os;
}

template<typename TX, typename TY, typename ...Others> std::ostream&
operator<<(std::ostream& os, std::map<TX, TY, Others...> const& d)
{
	return ContainerOutPut2(os, d.begin(), d.end());
}

template<typename TX, typename TY, typename ...Others> std::ostream&
operator<<(std::ostream& os, std::multimap<TX, TY, Others...> const& d)
{
	return ContainerOutPut2(os, d.begin(), d.end());
}

}  // namespace simpla

#endif /* PERTTY_STREAM_H_ */
