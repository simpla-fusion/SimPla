/**
 * @file lua_object_ext.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_UTILITIES_LUA_OBJECT_EXT_H_
#define CORE_UTILITIES_LUA_OBJECT_EXT_H_

#include <stddef.h>
#include <complex>
#include <list>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <simpla/algebra/nTuple.h>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

namespace simpla {
/** @ingroup toolbox */
template <typename T>
struct Converter {
    typedef T value_type;

    static unsigned int from(lua_State *L, unsigned int idx, value_type *v);

    static unsigned int to(lua_State *L, value_type const &v);
};

namespace _impl {
template <typename TC>
void push_container_to_lua(lua_State *L, TC const &v) {
    lua_newtable(L);

    size_t s = 1;
    for (auto const &vv : v) {
        lua_pushinteger(L, s);
        Converter<decltype(vv)>::to(L, vv);
        lua_settable(L, -3);
        ++s;
    }
}

inline unsigned int push_to_lua(lua_State *L) { return 0; }

template <typename T, typename... Args>
inline unsigned int push_to_lua(lua_State *L, T const &v, Args const &... rest) {
    luaL_checkstack(L, 1 + sizeof...(rest), "too many arguments");
    return Converter<T>::to(L, v) + push_to_lua(L, rest...);
}

inline unsigned int pop_from_lua(lua_State *L, int) { return 0; }

template <typename T, typename... Args>
inline unsigned int pop_from_lua(lua_State *L, int idx, T *v, Args *... rest) {
    return Converter<T>::from(L, idx, v) + pop_from_lua(L, idx + 1, rest...);
}
}  // namespace _impl

#define DEF_LUA_TRANS(_TYPE_, _TO_FUN_, _FROM_FUN_, _CHECK_FUN_)                                 \
    template <>                                                                                  \
    struct Converter<_TYPE_> {                                                                   \
        typedef _TYPE_ value_type;                                                               \
        static inline bool check(lua_State *L, unsigned int idx) { return _CHECK_FUN_(L, idx); } \
        static inline unsigned int from(lua_State *L, unsigned int idx, _TYPE_ *v) {             \
            if (!check(L, idx)) {                                                                \
                return 0;                                                                        \
            } else {                                                                             \
                *v = static_cast<_TYPE_>(_FROM_FUN_(L, idx));                                    \
                return 1;                                                                        \
            }                                                                                    \
        }                                                                                        \
        static inline unsigned int to(lua_State *L, _TYPE_ const &v) {                           \
            _TO_FUN_(L, v);                                                                      \
            return 1;                                                                            \
        }                                                                                        \
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
struct Converter<std::string> {
    typedef std::string value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (lua_isstring(L, idx)) {
            *v = lua_tostring(L, idx);
            return 1;
        }

        return 0;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        lua_pushstring(L, v.c_str());
        return 1;
    }
};

template <size_t N, typename T>
struct Converter<nTuple<T, N>> {
    typedef nTuple<T, N> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (lua_istable(L, idx)) {
            size_t num = lua_rawlen(L, idx);

            for (size_t s = 0; s < N; ++s) {
                lua_rawgeti(L, idx, static_cast<int>(s % num + 1));
                _impl::pop_from_lua(L, -1, &((*v)[s]));
                lua_pop(L, 1);
            }
            return 1;
        }

        return 0;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        lua_newtable(L);

        for (int i = 0; i < N; ++i) {
            lua_pushinteger(L, i + 1);
            Converter<T>::to(L, v[i]);
            lua_settable(L, -3);
        }
        return 1;
    }
};

// template<typename T, size_t  N, size_t  ...M> struct Converter<Tensor<T, N, M...>>
//{
//    typedef Tensor<T, N, M...> value_type;
//
//    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v)
//    {
//        if (lua_istable(L, idx))
//        {
//            size_t num = lua_rawlen(L, idx);
//            for (size_t s = 0; s < N; ++s)
//            {
//                lua_rawgeti(L, idx, static_cast<int>(s % num + 1));
//                _impl::pop_from_lua(L, -1, &((*v)[s]));
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
//    static inline unsigned int to(lua_State *L, value_type const &v)
//    {
//        lua_newtable(L);
//
//        for (int i = 0; i < N; ++i)
//        {
//            lua_pushinteger(L, i + 1);
//            Converter<T>::to(L, v[i]);
//            lua_settable(L, -3);
//        }
//        return 1;
//
//    }
//};

template <typename T>
struct Converter<std::vector<T>> {
    typedef std::vector<T> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (!lua_istable(L, idx)) { return 0; }
        size_t fnum = lua_rawlen(L, idx);
        if (fnum > 0) {
            for (size_t s = 0; s < fnum; ++s) {
                T res;
                lua_rawgeti(L, idx, static_cast<int>(s % fnum + 1));
                _impl::pop_from_lua(L, -1, &(res));
                lua_pop(L, 1);
                v->emplace_back(res);
            }
        }
        return 1;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        _impl::push_container_to_lua(L, v);
        return 1;
    }
};

template <typename T>
struct Converter<std::list<T>> {
    typedef std::list<T> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (!lua_istable(L, idx)) { return 0; }

        size_t fnum = lua_rawlen(L, idx);

        for (size_t s = 0; s < fnum; ++s) {
            lua_rawgeti(L, idx, static_cast<int>(s % fnum + 1));
            T tmp;
            _impl::pop_from_lua(L, -1, tmp);
            v->push_back(tmp);
            lua_pop(L, 1);
        }

        return 1;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        _impl::push_container_to_lua(L, v);
        return 1;
    }
};

template <typename T1, typename T2>
struct Converter<std::map<T1, T2>> {
    typedef std::map<T1, T2> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (!lua_istable(L, idx)) { return 0; }

        lua_pushnil(L); /* first key */

        T1 key;
        T2 value;

        while (lua_next(L, idx)) {
            /* uses 'key' (at index -2) and 'entity' (at index -1) */

            int top = lua_gettop(L);

            _impl::pop_from_lua(L, top - 1, &key);

            _impl::pop_from_lua(L, top, &value);
            (*v)[key] = value;
            /* removes 'entity'; keeps 'key' for next iteration */
            lua_pop(L, 1);
        }

        return 1;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        lua_newtable(L);

        for (auto const &vv : v) {
            Converter<T1>::to(L, vv.first);
            Converter<T2>::to(L, vv.second);
            lua_settable(L, -3);
        }
        return 1;
    }
};

template <typename T>
struct Converter<std::complex<T>> {
    typedef std::complex<T> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (lua_istable(L, idx)) {
            lua_pushnil(L); /* first key */
            while (lua_next(L, idx)) {
                /* uses 'key' (at index -2) and 'entity' (at index -1) */
                T r, i;
                _impl::pop_from_lua(L, -2, &r);
                _impl::pop_from_lua(L, -1, &i);
                /* removes 'entity'; keeps 'key' for next iteration */
                lua_pop(L, 1);

                *v = std::complex<T>(r, i);
            }

        } else if (lua_isnumber(L, idx)) {
            T r;
            _impl::pop_from_lua(L, idx, &r);
            *v = std::complex<T>(r, 0);
        } else {
            return 0;
        }
        return 1;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        Converter<int>::to(L, 0);
        Converter<T>::to(L, v.real());
        lua_settable(L, -3);
        Converter<int>::to(L, 1);
        Converter<T>::to(L, v.imag());
        lua_settable(L, -3);

        return 1;
    }
};

template <typename T1, typename T2>
struct Converter<std::pair<T1, T2>> {
    typedef std::pair<T1, T2> value_type;

    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        if (lua_istable(L, idx)) { return 0; }

        lua_pushnil(L); /* first key */
        while (lua_next(L, idx)) {
            /* uses 'key' (at index -2) and 'entity' (at index -1) */

            _impl::pop_from_lua(L, -2, &(v->first));
            _impl::pop_from_lua(L, -1, &(v->second));
            /* removes 'entity'; keeps 'key' for next iteration */
            lua_pop(L, 1);
        }

        return 1;
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        Converter<T1>::to(L, v.first);
        Converter<T2>::to(L, v.second);
        lua_settable(L, -3);
        return 1;
    }
};

template <typename... T>
struct Converter<std::tuple<T...>> {
    typedef std::tuple<T...> value_type;

   private:
    static inline unsigned int from_(lua_State *L, unsigned int idx, value_type *v,
                                     std::integral_constant<unsigned int, 0>) {
        return 0;
    }

    template <unsigned int N>
    static inline unsigned int from_(lua_State *L, unsigned int idx, value_type *v,
                                     std::integral_constant<unsigned int, N>) {
        lua_rawgeti(L, idx, N);                                       // lua table's index starts from 1
        auto num = _impl::pop_from_lua(L, -1, &std::get<N - 1>(*v));  // C++ tuple index start from 0
        lua_pop(L, 1);

        return num + from_(L, idx, v, std::integral_constant<unsigned int, N - 1>());
    }

    static inline unsigned int to_(lua_State *L, value_type const &v, std::integral_constant<unsigned int, 0>) {
        return 0;
    }

    template <unsigned int N>
    static inline unsigned int to_(lua_State *L, value_type const &v, std::integral_constant<unsigned int, N>) {
        return _impl::push_to_lua(L, std::get<sizeof...(T) - N>(v)) +
               to_(L, v, std::integral_constant<unsigned int, N - 1>());
    }

   public:
    static inline unsigned int from(lua_State *L, unsigned int idx, value_type *v) {
        return from_(L, idx, v, std::integral_constant<unsigned int, sizeof...(T)>());
    }

    static inline unsigned int to(lua_State *L, value_type const &v) {
        return to_(L, v, std::integral_constant<unsigned int, sizeof...(T)>());
    }
};

/** @} LuaTrans */

}  // namespace simpla

#endif /* CORE_UTILITIES_LUA_OBJECT_EXT_H_ */
