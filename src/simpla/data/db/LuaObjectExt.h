/**
 * @file LuaObjectExt.h
 *
 * @date 2015-6-10
 * @author salmon
 */
#ifndef CORE_UTILITIES_LUA_OBJECT_EXT_H_
#define CORE_UTILITIES_LUA_OBJECT_EXT_H_

#include <complex>
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "simpla/algebra/nTuple.h"

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

namespace simpla {
/** @ingroup toolbox */
template <typename T>
struct LuaConverter {
    typedef T value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        return 0;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        return 0;
    }
};
template <typename T>
int push_to_lua(lua_State *L, T const *v, size_type ndims = 0, size_type const *extents = nullptr) {
    return LuaConverter<T>::to(L, v, ndims, extents);
}
template <typename T>
int push_to_lua(lua_State *L, T const &v) {
    return LuaConverter<T>::to(L, &v, 0, nullptr);
}
template <typename T>
int pop_from_lua(lua_State *L, int idx, T *v, size_type *rank = nullptr, size_type *extents = nullptr) {
    return LuaConverter<T>::from(L, idx, v, rank, extents);
}

template <typename TC>
void push_container_to_lua(lua_State *L, TC const &v) {
    lua_newtable(L);

    int s = 1;
    for (auto const &vv : v) {
        lua_pushinteger(L, s);
        LuaConverter<decltype(vv)>::to(L, vv);
        lua_settable(L, -3);
        ++s;
    }
}

//    template <>
//    struct LuaConverter<double> {
//        typedef double value_type;
//        static int from(lua_State *L, int idx, double *v, size_type *rank, size_type *extents) {
//            int count = 0;
//            if (lua_isnumber(L, idx) > 0) {
//                if (v != nullptr) { *v = lua_tonumber(L, idx); }
//                count = 1;
//            }
//
//            //        else if (lua_istable(L, idx)) {
//            //            size_t len = lua_rawlen(L, idx);
//            //            extents[0] = std::max(extents[0], len);
//            //            *rank += 1;
//            //            ASSERT(*rank < MAX_NDIMS_OF_ARRAY);
//            //            for (int i = 1; i <= len; ++i) {
//            //                lua_rawgeti(L, idx, i);
//            //                count += from(L, lua_gettop(L), v == nullptr ? nullptr : v + count, rank, extents + 1);
//            //                lua_pop(L, 1);
//            //            }
//            //        } else {
//            //        }
//
//            return count;
//        }
//        static int to(lua_State *L, double const *v, size_type ndims = 0, size_type const *extents = nullptr) {
//            int count = 0;
//            if (ndims == 0 || extents == nullptr) {
//                lua_pushnumber(L, *v);
//                count = 1;
//            }
//            //        else {
//            //            lua_newtable(L);
//            //            for (size_type i = 0; i < extents[0]; ++i) {
//            //                lua_pushinteger(L, static_cast<int>(i + 1));
//            //                count += to(L, v + count, ndims - 1, extents + 1);
//            //                lua_settable(L, -3);
//            //            }
//            //        }
//            return count;
//        }
//    };
#define DEF_LUA_TRANS(_TYPE_, _TO_FUN_, _FROM_FUN_, _CHECK_FUN_)                                                      \
    template <>                                                                                                       \
    struct LuaConverter<_TYPE_> {                                                                                     \
        typedef _TYPE_ value_type;                                                                                    \
        static int from(lua_State *L, int idx, _TYPE_ *v, size_type *ndims = nullptr, size_type *extents = nullptr) { \
            int count = 0;                                                                                            \
            if (_CHECK_FUN_(L, idx)) {                                                                                \
                *v = static_cast<_TYPE_>(_FROM_FUN_(L, idx));                                                         \
                count = 1;                                                                                            \
            }                                                                                                         \
            return count;                                                                                             \
        }                                                                                                             \
        static int to(lua_State *L, _TYPE_ const *v, size_type ndims = 0, size_type const *extents = nullptr) {       \
            _TO_FUN_(L, *v);                                                                                          \
            return 1;                                                                                                 \
        }                                                                                                             \
    };

typedef unsigned long ulong;
typedef unsigned int uint;

DEF_LUA_TRANS(float, lua_pushnumber, lua_tonumber, lua_isnumber)
DEF_LUA_TRANS(double, lua_pushnumber, lua_tonumber, lua_isnumber)
DEF_LUA_TRANS(int, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(long, lua_pushinteger, lua_tointeger, lua_isnumber)
// DEF_LUA_TRANS(ulong, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(size_t, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(uint, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(bool, lua_pushboolean, lua_toboolean, lua_isboolean)

#undef DEF_LUA_TRANS

template <>
struct LuaConverter<std::string> {
    typedef std::string value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (lua_isstring(L, idx) > 0) {
            *v = lua_tostring(L, idx);
            return 1;
        }
        return 0;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        lua_pushstring(L, v->c_str());
        return 1;
    }
};
template <typename T, int... N>
struct nTuple;
template <typename T, int... N>
struct LuaConverter<nTuple<T, N...>> {
    typedef T value_type;
    typedef nTuple<T, N...> second_type;
    static int from(lua_State *L, int idx, second_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (lua_istable(L, idx)) {
            size_t num = lua_rawlen(L, idx);

            //            for (size_t s = 0; s < N; ++s) {
            //                lua_rawgeti(L, idx, static_cast<int>(s % num + 1));
            //                pop_from_lua(L, -1, &((*v)[s]));
            //                lua_pop(L, 1);
            //            }
            return 1;
        }

        return 0;
    }

    static int to(lua_State *L, second_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        lua_newtable(L);

        //        for (int i = 0; i < N; ++i) {
        //            lua_pushinteger(L, i + 1);
        //            LuaConverter<T>::to(L, v[i]);
        //            lua_settable(L, -3);
        //        }
        return 1;
    }
};

// template<typename T, size_t  N, size_t  ...M> struct LuaConverter<Tensor<T, N, M...>>
//{
//    typedef Tensor<T, N, M...> value_type;
//
//    static inline   int from(lua_State *L,   int idx, value_type *v)
//    {
//        if (lua_istable(L, idx))
//        {
//            size_t num = lua_rawlen(L, idx);
//            for (size_t s = 0; s < N; ++s)
//            {
//                lua_rawgeti(L, idx, static_cast<int>(s % num + 1));
//                pop_from_lua(L, -1, &((*v)[s]));
//                lua_pop(L, 1);
//            }
//
//            return 1;
//
//        }
//
//        return 0;
//    }
//
//    static inline   int to(lua_State *L, value_type const &v)
//    {
//        lua_newtable(L);
//
//        for (int i = 0; i < N; ++i)
//        {
//            lua_pushinteger(L, i + 1);
//            LuaConverter<T>::to(L, v[i]);
//            lua_settable(L, -3);
//        }
//        return 1;
//
//    }
//};

template <typename T>
struct LuaConverter<std::vector<T>> {
    typedef std::vector<T> value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (!lua_istable(L, idx)) { return 0; }
        size_t fnum = lua_rawlen(L, idx);
        if (fnum > 0) {
            for (size_t s = 0; s < fnum; ++s) {
                T res;
                lua_rawgeti(L, idx, static_cast<int>(s % fnum + 1));
                pop_from_lua(L, -1, &(res));
                lua_pop(L, 1);
                v->emplace_back(res);
            }
        }
        return 1;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        push_container_to_lua(L, v);
        return 1;
    }
};

template <typename T>
struct LuaConverter<std::list<T>> {
    typedef std::list<T> value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (!lua_istable(L, idx)) { return 0; }

        size_t fnum = lua_rawlen(L, idx);

        for (size_t s = 0; s < fnum; ++s) {
            lua_rawgeti(L, idx, static_cast<int>(s % fnum + 1));
            T tmp;
            pop_from_lua(L, -1, tmp);
            v->push_back(tmp);
            lua_pop(L, 1);
        }

        return 1;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        push_container_to_lua(L, v);
        return 1;
    }
};

template <typename T1, typename T2>
struct LuaConverter<std::map<T1, T2>> {
    typedef std::map<T1, T2> value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (!lua_istable(L, idx)) { return 0; }

        lua_pushnil(L); /* first key */

        T1 key;
        T2 value;

        while (lua_next(L, idx) > 0) {
            /* uses 'key' (at index -2) and 'entity' (at index -1) */

            int top = lua_gettop(L);
            pop_from_lua(L, top - 1, &key);
            pop_from_lua(L, top, &value);
            (*v)[key] = value;
            /* removes 'entity'; keeps 'key' for next iteration */
            lua_pop(L, 1);
        }

        return 1;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        lua_newtable(L);

        for (auto const &vv : *v) {
            push_to_lua<T1>(L, vv.first);
            push_to_lua<T2>(L, vv.second);
            lua_settable(L, -3);
        }
        return 1;
    }
};

template <typename T>
struct LuaConverter<std::complex<T>> {
    typedef std::complex<T> value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (lua_istable(L, idx)) {
            lua_pushnil(L); /* first key */
            while (lua_next(L, idx)) {
                /* uses 'key' (at index -2) and 'entity' (at index -1) */
                T r, i;
                pop_from_lua(L, -2, &r);
                pop_from_lua(L, -1, &i);
                /* removes 'entity'; keeps 'key' for next iteration */
                lua_pop(L, 1);

                *v = std::complex<T>(r, i);
            }

        } else if (lua_isnumber(L, idx)) {
            T r;
            pop_from_lua(L, idx, &r);
            *v = std::complex<T>(r, 0);
        } else {
            return 0;
        }
        return 1;
    }

    static int to(lua_State *L, value_type const &v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        push_to_lua<int>(L, 0);
        push_to_lua<T>(L, v.real());
        lua_settable(L, -3);
        push_to_lua<int>(L, 1);
        push_to_lua<T>(L, v.imag());
        lua_settable(L, -3);

        return 1;
    }
};

template <typename T1, typename T2>
struct LuaConverter<std::pair<T1, T2>> {
    typedef std::pair<T1, T2> value_type;

    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr, size_type *extents = nullptr) {
        if (lua_istable(L, idx)) { return 0; }

        lua_pushnil(L); /* first key */
        while (lua_next(L, idx) > 0) {
            /* uses 'key' (at index -2) and 'entity' (at index -1) */

            pop_from_lua(L, -2, &(v->first));
            pop_from_lua(L, -1, &(v->second));
            /* removes 'entity'; keeps 'key' for next iteration */
            lua_pop(L, 1);
        }

        return 1;
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        LuaConverter<T1>::to(L, v->first);
        LuaConverter<T2>::to(L, v->second);
        lua_settable(L, -3);
        return 1;
    }
};

template <typename... T>
struct LuaConverter<std::tuple<T...>> {
    typedef std::tuple<T...> value_type;

   private:
    static int from_(lua_State *L, int idx, value_type *v, std::integral_constant<int, 0>) { return 0; }

    template <int N>
    static int from_(lua_State *L, int idx, value_type *v, std::integral_constant<int, N>) {
        lua_rawgeti(L, idx, N);                                // lua table's index starts from 1
        auto num = pop_from_lua(L, -1, &std::get<N - 1>(*v));  // C++ tuple index start from 0
        lua_pop(L, 1);

        return num + from_(L, idx, v, std::integral_constant<int, N - 1>());
    }

    static int to_(lua_State *L, value_type const &v, std::integral_constant<int, 0>) { return 0; }

    template <int N>
    static int to_(lua_State *L, value_type const &v, std::integral_constant<int, N>) {
        return push_to_lua(L, std::get<sizeof...(T) - N>(v)) + to_(L, v, std::integral_constant<int, N - 1>());
    }

   public:
    static int from(lua_State *L, int idx, value_type *v, size_type *ndims = nullptr,
                    size_type const *extents = nullptr) {
        return from_(L, idx, v, std::integral_constant<int, sizeof...(T)>());
    }

    static int to(lua_State *L, value_type const *v, size_type ndims = 0, size_type const *extents = nullptr) {
        return to_(L, v, std::integral_constant<int, sizeof...(T)>());
    }
};

/** @} LuaTrans */

}  // namespace simpla

#endif /* CORE_UTILITIES_LUA_OBJECT_EXT_H_ */
