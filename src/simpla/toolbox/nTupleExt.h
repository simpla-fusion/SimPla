/**
 * @file ntuple_ext.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_toolbox_NTUPLE_EXT_H_
#define CORE_toolbox_NTUPLE_EXT_H_

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>

#include "nTuple.h"
#include "type_cast.h"

namespace simpla
{
template<typename, size_t...> struct nTuple;


template<typename T, size_t N> std::istream &
operator>>(std::istream &is, nTuple<T, N> &tv)
{
    for (size_t i = 0; i < N && is; ++i)
    {
        is >> tv[i];
    }

    return (is);
}

template<typename T, size_t M>
std::ostream &operator<<(std::ostream &os, nTuple<T, M> const &v)
{
    os << "{" << v[0];
    for (size_t i = 1; i < M; ++i)
    {
        os << " , " << v[i];
    }
    os << "}";

    return os;
}


template<typename T, size_t M, size_t M2, size_t ...N>
std::ostream &operator<<(std::ostream &os, nTuple<T, M, M2, N...> const &v)
{
    os << "{" << v[0];
    for (size_t i = 1; i < M; ++i)
    {
        os << " , " << v[i];
    }
    os << "}";

    return os;
}

namespace traits
{
template<typename TSrc, typename TDesc> struct type_cast;

template<size_t N, typename T>
struct type_cast<nTuple<T, N>, std::string>
{
    static std::string eval(nTuple<T, N> const &v)
    {
        std::ostringstream buffer;
        buffer << v;
        return buffer.str();
    }
};

template<size_t N, typename T>
struct type_cast<std::string, nTuple<T, N> >
{
    static nTuple<T, N> eval(std::string const &s)
    {
        nTuple<T, N> v;
        std::istringstream is(s);
        is >> v;
        return std::move(v);

    }
};
//
//	template<typename T>
//	typename array_to_ntuple_convert<T>::type as(T const & default_v) const
//	{
//		typename array_to_ntuple_convert<T>::type res = default_v;
//		if (!value_type::empty())
//		{
//			res = value_type::template as<
//					typename array_to_ntuple_convert<T>::type>();
//		}
//
//		return std::move(res);
//	}
} // namespace traits
}  // namespace simpla
//
//namespace std
//{
//
//template<size_t M, typename T, size_t N>
//T const & get(simpla::nTuple<T, N> const & v)
//{
//	return v[M];
//}
///**
// * C++11 <type_traits>
// * @ref http://en.cppreference.com/w/cpp/types/rank
// */
//template<typename T, size_t ...N>
//struct rank<simpla::nTuple<T, N...>> : public std::size_tegral_constant<
//		std::size_t, sizeof...(N)>
//{
//};
//
///**
// * C++11 <type_traits>
// * @ref http://en.cppreference.com/w/cpp/types/extent
// */
//
//template<class T, std::size_t N, std::size_t ...M>
//struct extent<simpla::nTuple<T, N, M...>, 0> : std::integral_constant<
//		std::int, N>
//{
//};
//
//template<std::int I, class T, std::int N, std::int ...M>
//struct extent<simpla::nTuple<T, N, M...>, I> : public std::integral_constant<
//		std::int, std::extent<simpla::nTuple<T, M...>, I - 1>::entity>
//{
//};
//
///**
// * C++11 <type_traits>
// * @ref http://en.cppreference.com/w/cpp/types/remove_all_extents
// */
//template<class T, std::int ...M>
//struct remove_all_extents<simpla::nTuple<T, M...> >
//{
//	typedef T type;
//};
//
////template<typename T, int I>
////class std::less<simpla::nTuple<T, I> >
////{
////public:
////	bool operator()(const simpla::nTuple<T, I>& x,
////			const simpla::nTuple<T, I>& y) const
////	{
////		for (int i = 0; i < I; ++i)
////		{
////			if (x[i] < y[i])
////				return true;
////		}
////		return false;
////	}
////};
//}// namespace std
#endif /* CORE_toolbox_NTUPLE_EXT_H_ */
