/**
 * @file lua_object_ext.h
 *
 * @date 2015-6-10
 * @author salmon
 */

#ifndef CORE_UTILITIES_LUA_OBJECT_EXT_H_
#define CORE_UTILITIES_LUA_OBJECT_EXT_H_

#include <lua.h>
#include <stddef.h>
#include <complex>
#include <list>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../gtl/ntuple.h"
#include "lua_object.h"

namespace simpla
{
namespace lua
{

template<unsigned int N, typename T> struct Converter<nTuple<T, N>>
{
	typedef nTuple<T, N> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			size_t num = lua_rawlen(L.get(), idx);
			for (size_t s = 0; s < N; ++s)
			{
				lua_rawgeti(L.get(), idx, s % num + 1);
				_impl::pop_from_lua(L, -1, &((*v)[s]));
				lua_pop(L.get(), 1);
			}

		}
		else
		{
			*v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		lua_newtable(L.get());

		for (int i = 0; i < N; ++i)
		{
			lua_pushinteger(L.get(), i + 1);
			Converter<T>::to(L, v[i]);
			lua_settable(L.get(), -3);
		}
		return 1;

	}
};

template<typename T, size_t N, size_t ...M> struct Converter<nTuple<T, N, M...>>
{
	typedef nTuple<T, N, M...> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			size_t num = lua_rawlen(L.get(), idx);
			for (size_t s = 0; s < N; ++s)
			{
				lua_rawgeti(L.get(), idx, s % num + 1);
				_impl::pop_from_lua(L, -1, &((*v)[s]));
				lua_pop(L.get(), 1);
			}

		}
		else
		{
			*v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		lua_newtable(L.get());

		for (int i = 0; i < N; ++i)
		{
			lua_pushinteger(L.get(), i + 1);
			Converter<T>::to(L, v[i]);
			lua_settable(L.get(), -3);
		}
		return 1;

	}
};

template<typename T> struct Converter<std::vector<T> >
{
	typedef std::vector<T> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			size_t fnum = lua_rawlen(L.get(), idx);

			if (fnum > 0)
			{

				for (size_t s = 0; s < fnum; ++s)
				{
					T res;
					lua_rawgeti(L.get(), idx, s % fnum + 1);
					_impl::pop_from_lua(L, -1, &(res));
					lua_pop(L.get(), 1);
					v->emplace_back(res);
				}
			}

		}
		else
		{
			*v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		_impl::push_container_to_lua(L, v);
		return 1;
	}
};
template<typename T> struct Converter<std::list<T> >
{
	typedef std::list<T> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			size_t fnum = lua_rawlen(L.get(), idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(L.get(), idx, s % fnum + 1);
				T tmp;
				_impl::pop_from_lua(L, -1, tmp);
				v->push_back(tmp);
				lua_pop(L.get(), 1);
			}

		}
		else
		{
			v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		_impl::push_container_to_lua(L, v);
		return 1;
	}
};

template<typename T1, typename T2> struct Converter<std::map<T1, T2> >
{
	typedef std::map<T1, T2> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			lua_pushnil(L.get()); /* first key */

			T1 key;
			T2 value;

			while (lua_next(L.get(), idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				_impl::pop_from_lua(L, -2, &key);
				_impl::pop_from_lua(L, -1, &value);
				(*v)[key] = value;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(L.get(), 1);
			}

		}
		else
		{
			v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		lua_newtable(L.get());

		for (auto const & vv : v)
		{
			Converter<T1>::to(L, vv.first);
			Converter<T2>::to(L, vv.second);
			lua_settable(L.get(), -3);
		}
		return 1;
	}
};

template<typename T> struct Converter<std::complex<T> >
{
	typedef std::complex<T> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			lua_pushnil(L.get()); /* first key */
			while (lua_next(L.get(), idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */
				T r, i;
				_impl::pop_from_lua(L, -2, &r);
				_impl::pop_from_lua(L, -1, &i);
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(L.get(), 1);

				*v = std::complex<T>(r, i);
			}

		}
		else if (lua_isnumber(L.get(), idx))
		{
			T r;
			_impl::pop_from_lua(L, idx, &r);
			*v = std::complex<T>(r, 0);
		}
		else
		{
			*v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		Converter<int>::to(L, 0);
		Converter<T>::to(L, v.real());
		lua_settable(L.get(), -3);
		Converter<int>::to(L, 1);
		Converter<T>::to(L, v.imag());
		lua_settable(L.get(), -3);

		return 1;
	}
};

template<typename T1, typename T2> struct Converter<std::pair<T1, T2> >
{
	typedef std::pair<T1, T2> value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type * v, value_type const &default_value =
					value_type())
	{
		if (lua_istable(L.get(), idx))
		{
			lua_pushnil(L.get()); /* first key */
			while (lua_next(L.get(), idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				_impl::pop_from_lua(L, -2, &(v->first));
				_impl::pop_from_lua(L, -1, &(v->second));
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(L.get(), 1);
			}

		}
		else
		{
			*v = default_value;
		}
		return 1;
	}
	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const & v)
	{
		Converter<T1>::to(L, v.first);
		Converter<T2>::to(L, v.second);
		lua_settable(L.get(), -3);
		return 1;
	}
};

template<typename ... T> struct Converter<std::tuple<T...> >
{
	typedef std::tuple<T...> value_type;

private:
	static inline unsigned int from_(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type* v,
			std::integral_constant<unsigned int, 0>)
	{
		return 0;
	}

	template<unsigned int N>
	static inline unsigned int from_(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type* v,
			std::integral_constant<unsigned int, N>)
	{
		lua_rawgeti(L.get(), idx, N); // lua table's index starts from 1
		auto num = _impl::pop_from_lua(L, -1, &std::get<N - 1>(*v)); // C++ tuple index start from 0
		lua_pop(L.get(), 1);

		return num
				+ from_(L, idx, v,
						std::integral_constant<unsigned int, N - 1>());
	}
	static inline unsigned int to_(std::shared_ptr<lua_State> L,
			value_type const& v, std::integral_constant<unsigned int, 0>)
	{
		return 0;
	}
	template<unsigned int N> static inline unsigned int to_(
			std::shared_ptr<lua_State> L, value_type const& v,
			std::integral_constant<unsigned int, N>)
	{
		return _impl::push_to_lua(L, std::get<sizeof...(T) - N >(v))
		+ to_(L, v, std::integral_constant< unsigned int , N - 1>());
	}
public:
	static inline unsigned int from(std::shared_ptr<lua_State> L,
	unsigned int idx, value_type * v, value_type const &default_value =
	value_type())
	{
		return from_(L, idx, v, std::integral_constant< unsigned int , sizeof...(T)>());

	}
	static inline unsigned int to(std::shared_ptr<lua_State> L, value_type const & v)
	{
		return to_(L, v,std::integral_constant< unsigned int , sizeof...(T)>());
	}

};

} // namespace lua
		}																		 // namespace simpla

#endif /* CORE_UTILITIES_LUA_OBJECT_EXT_H_ */
