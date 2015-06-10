/**
 * @file ntuple_ext.h
 *
 * @date 2015年6月10日
 * @author salmon
 */

#ifndef CORE_GTL_NTUPLE_EXT_H_
#define CORE_GTL_NTUPLE_EXT_H_

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include "ntuple.h"
#include "type_cast.h"

namespace simpla
{
template<typename T, size_t N> std::istream &
operator>>(std::istream& is, nTuple<T, N> & tv)
{
	for (int i = 0; i < N && is; ++i)
	{
		is >> tv[i];
	}

	return (is);
}
template<typename T, size_t M>
std::ostream &operator<<(std::ostream & os, nTuple<T, M> const & v)
{
	os << "{" << v[0];
	for (int i = 1; i < M; ++i)
	{
		os << " , " << v[i];
	}
	os << "}";

	return os;
}

template<typename T, size_t M, size_t M2, size_t ...N>
std::ostream &operator<<(std::ostream & os, nTuple<T, M, M2, N...> const & v)
{
	os << "{" << v[0];
	for (int i = 1; i < M; ++i)
	{
		os << " , " << v[i] << std::endl;
	}
	os << "}" << std::endl;

	return os;
}

namespace traits
{
template<typename TSrc, typename TDesc> struct type_cast;

template<unsigned int N, typename T>
struct type_cast<nTuple<T, N>, std::string>
{
	static std::string eval(nTuple<T, N> const &v)
	{
		std::ostringstream buffer;
		buffer << v;
		return buffer.str();
	}
};

template<unsigned int N, typename T>
struct type_cast<std::string, nTuple<T, N>>
{
	static nTuple<T, N> eval(std::string const &s)
	{
		nTuple<T, N> v;
		std::istringstream is(s);
		is >> v;
		return std::move(v);

	}
};

} // namespace traits
}  // namespace simpla

namespace std
{

template<size_t M, typename T, size_t N>
T const & get(simpla::nTuple<T, N> const & v)
{
	return v[M];
}
/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/rank
 */
template<typename T, size_t ...N>
struct rank<simpla::nTuple<T, N...>> : public std::integral_constant<
		std::size_t, sizeof...(N)>
{
};

/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/extent
 */

template<class T, std::size_t N, std::size_t ...M>
struct extent<simpla::nTuple<T, N, M...>, 0> : std::integral_constant<
		std::size_t, N>
{
};

template<std::size_t I, class T, std::size_t N, std::size_t ...M>
struct extent<simpla::nTuple<T, N, M...>, I> : public std::integral_constant<
		std::size_t, std::extent<simpla::nTuple<T, M...>, I - 1>::value>
{
};

/**
 * C++11 <type_traits>
 * @ref http://en.cppreference.com/w/cpp/types/remove_all_extents
 */
template<class T, std::size_t ...M>
struct remove_all_extents<simpla::nTuple<T, M...> >
{
	typedef T type;
};

//template<typename T, size_t I>
//class std::less<simpla::nTuple<T, I> >
//{
//public:
//	bool operator()(const simpla::nTuple<T, I>& x,
//			const simpla::nTuple<T, I>& y) const
//	{
//		for (int i = 0; i < I; ++i)
//		{
//			if (x[i] < y[i])
//				return true;
//		}
//		return false;
//	}
//};
}// namespace std
#endif /* CORE_GTL_NTUPLE_EXT_H_ */
