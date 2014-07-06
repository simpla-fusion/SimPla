/*
 * pretty_stream.h
 *
 *  Created on: 2013年11月29日
 *      Author: salmon
 */

#ifndef PRETTY_STREAM_H_
#define PRETTY_STREAM_H_

#include <complex>

#include <iterator>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ntuple.h"
#include "sp_type_traits.h"

namespace simpla
{

//! \ingroup Utilities

template<typename TV> inline TV const *
PrintNdArray(std::ostream & os, TV const *v, int rank, size_t const* d, std::string const & left_brace = "{",
        std::string const & sep = ",", std::string const & right_brace = "}")
{
	if (rank == 1)
	{
		os << left_brace << *v;
		++v;
		for (int s = 1; s < d[0]; ++s)
		{
			os << sep << "\t";
			if (s % 5 == 0)
				os << std::endl;
			os << (*v);
			++v;
		}
		os << right_brace << std::endl;
		return v;
	}
	else
	{

		os << left_brace;
		v = PrintNdArray(os, v, rank - 1, d + 1, left_brace, sep, right_brace);

		for (int s = 1; s < d[0]; ++s)
		{
			os << sep << std::endl;
			v = PrintNdArray(os, v, rank - 1, d + 1, left_brace, sep, right_brace);
		}
		os << right_brace << std::endl;
		return (v);
	}
}

template<int N, typename T> std::ostream &
operator<<(std::ostream& os, const nTuple<N, T> & tv)
{
	os << "{" << tv[0];
	for (int i = 1; i < N; ++i)
	{
		os << "," << tv[i];
	}
	os << "}";
	return (os);
}
template<int N, typename T> std::istream &
operator>>(std::istream& is, nTuple<N, T> & tv)
{
	for (int i = 0; i < N && is; ++i)
	{
		is >> tv[i];
	}

	return (is);
}

template<int N, typename T> nTuple<N, T> ToNTuple(std::string const & str)
{
	std::istringstream ss(str);
	nTuple<N, T> res;
	ss >> res;
	return (res);
}
template<typename TX, typename TY, typename ...Others> std::istream&
get_(std::istream& is, size_t num, std::map<TX, TY, Others...> & a)
{

	for (size_t s = 0; s < num; ++s)
	{
		TX x;
		TY y;
		is >> x >> y;
		a.emplace(x, y);
	}
	return is;
}

template<typename TI>
std::ostream & ContainerOutPut1(std::ostream & os, TI const ib, TI const ie)
{
	if (ib == ie)
		return os;

	TI it = ib;

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

template<typename TI>
std::ostream & ContainerOutPut2(std::ostream & os, TI const & ib, TI const & ie)
{
	if (ib == ie)
		return os;

	TI it = ib;

	os << it->first << "=" << it->second;

	++it;

	for (; it != ie; ++it)
	{
		os << " , " << it->first << " = " << it->second << "\n";
	}
	return os;
}
template<typename TI, typename TFUN>
std::ostream & ContainerOutPut3(std::ostream & os, TI const & ib, TI const & ie, TFUN const& fun)
{
	if (ib == ie)
		return os;

	TI it = ib;

	fun(os, it);

	++it;

	for (; it != ie; ++it)
	{
		os << " , ";
		fun(os, it);
	}
	return os;
}
} // namespace simpla

namespace std
{

template<typename T> std::ostream &
operator<<(std::ostream& os, const std::complex<T> & tv)
{
	os << "{" << tv.real() << "," << tv.imag() << "}";
	return (os);
}
template<typename T1, typename T2> std::ostream &
operator<<(std::ostream& os, std::pair<T1, T2>const & p)
{
	os << p.first << " = { " << p.second << " }";
	return (os);
}
template<typename T1, typename T2> std::ostream &
operator<<(std::ostream& os, std::map<T1, T2>const & p)
{
	for (auto const & v : p)
		os << v << "," << std::endl;
	return (os);
}

template<typename TV, typename ...Others> std::istream&
operator>>(std::istream& is, std::vector<TV, Others...> & a)
{
	for (auto & v : a)
	{
		is >> v;
	}
//	std::copy(std::istream_iterator<TV>(is), std::istream_iterator<TV>(), std::back_inserter(a));
	return is;

}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::vector<U, Others...> const &d)
{
	return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::list<U, Others...> const &d)
{
	return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::set<U, Others...> const &d)
{
	return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template<typename U, typename ...Others>
std::ostream & operator<<(std::ostream & os, std::multiset<U, Others...> const &d)
{
	return simpla::ContainerOutPut1(os, d.begin(), d.end());
}

template<typename TX, typename TY, typename ...Others> std::ostream&
operator<<(std::ostream& os, std::map<TX, TY, Others...> const& d)
{
	return simpla::ContainerOutPut2(os, d.begin(), d.end());
}

template<typename TX, typename TY, typename ...Others> std::ostream&
operator<<(std::ostream& os, std::multimap<TX, TY, Others...> const& d)
{
	return simpla::ContainerOutPut2(os, d.begin(), d.end());
}

}  // namespace std

//namespace _impl
//{
//
//HAS_MEMBER_FUNCTION(Serialize)
//
//}  // namespace _impl
//
//template<typename TOS, typename TD>
//auto PrettyOutput(TOS & os, TD const &d)
//ENABLE_IF_DECL_RET_TYPE( (_impl::has_member_function_Serialize<TD,TOS>::value), d.Serialize(os))
//
//template<typename TOS, typename TD>
//auto PrettyOutput(TOS & os, TD const &d)
//ENABLE_IF_DECL_RET_TYPE( (!_impl::has_member_function_Serialize<TD,TOS>::value), Serialize(os,d))
//
//template<typename TOS, typename TD>
//TOS & operator<<(TOS & os, TD const &d)
//{
//	return PrettyOutput(os, d);
//}

#endif /* PRETTY_STREAM_H_ */
