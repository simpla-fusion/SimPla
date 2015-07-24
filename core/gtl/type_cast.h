/**
 * @file type_cast.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_GTL_TYPE_CAST_H_
#define CORE_GTL_TYPE_CAST_H_

#include <sstream>
#include <string>

namespace simpla {
namespace gtl {
namespace traits {

template<typename TSrc, typename TDesc>
struct type_cast
{
    static constexpr TDesc eval(TSrc const &v)
    {
        return static_cast<TDesc>(v);
    }
};

template<typename TSrc>
struct type_cast<TSrc, std::string>
{
    static std::string eval(TSrc const &v)
    {
        std::ostringstream buffer;
        buffer << v;
        return buffer.str();
    }
};

template<typename T>
struct type_cast<std::string, T>
{
    static T eval(std::string const &s)
    {
        T v;
        std::istringstream is(s);
        is >> v;
        return std::move(v);
    }
};

template<>
struct type_cast<std::string, std::string>
{
    static std::string const &eval(std::string const &v)
    {
        return v;
    }
};

template<typename TSrc>
struct type_cast<TSrc, TSrc>
{
    static TSrc const &eval(TSrc const &v)
    {
        return v;
    }
};
}  // namespace traits

template<typename T, typename U>
T &raw_cast(U &s)
{
    return *reinterpret_cast<T *>(&s);
}

template<typename T, typename U>
T raw_cast(U &&s)
{
    return *reinterpret_cast<T *>(&s);
}

template<typename T, typename U>
T assign_cast(U const &s)
{
    T res;
    res = s;
    return std::move(res);
}

template<typename TDest, typename TSrc>
TDest type_cast(TSrc const &v)
{
    return traits::type_cast<TSrc, TDest>::eval(v);
}
}
}//  namespace simpla::gtl

#endif /* CORE_GTL_TYPE_CAST_H_ */
