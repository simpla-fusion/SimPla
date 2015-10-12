/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * @file lua_object.h
 *
 * @date 2010-9-22
 * @author salmon
 */

#ifndef CORE_UTILITIES_LUA_OBJECT_H_
#define CORE_UTILITIES_LUA_OBJECT_H_

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "log.h"
#include "../type_cast.h"

extern "C"
{

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#if LUA_VERSION_NUM < 502
#error  need lua version >502
#endif
}

namespace simpla
{

namespace lua
{

/**
 * @ingroup utilities
 * @addtogroup  lua   Lua engine
 *  @{
 */

#define LUA_ERROR(_CMD_)                                                     \
{                                                                              \
   int error=_CMD_;                                                            \
    if(error!=0)                                                               \
    {                                                                          \
     logger::Logger(logger::LOG_ERROR)                                                         \
      <<"["<<__FILE__<<":"<<__LINE__<<":"<<  (__PRETTY_FUNCTION__)<<"]:"       \
      << lua_tostring(L_.get(), -1)<<std::endl ;                               \
     lua_pop(L_.get(), 1);                                                     \
     throw(std::runtime_error("Lua error"));                                   \
    }                                                                          \
}

class Object;

template<typename T> struct Converter;

namespace _impl
{
template<typename TC> void push_container_to_lua(std::shared_ptr<lua_State> L,
		TC const &v)
{
	lua_newtable(L.get());

	size_t s = 1;
	for (auto const &vv : v)
	{
		lua_pushinteger(L.get(), s);
		Converter<decltype(vv)>::to(L, vv);
		lua_settable(L.get(), -3);
		++s;
	}
}

inline unsigned int push_to_lua(std::shared_ptr<lua_State> L)
{
	return 0;
}

template<typename T, typename ... Args>
inline unsigned int push_to_lua(std::shared_ptr<lua_State> L, T const &v,
		Args const &... rest)
{
	luaL_checkstack(L.get(), 1 + sizeof...(rest), "too many arguments");

	return Converter<T>::to(L, v) + push_to_lua(L, rest...);
}

inline unsigned int pop_from_lua(std::shared_ptr<lua_State> L, int)
{
	return 0;
}

template<typename T, typename ... Args>
inline unsigned int pop_from_lua(std::shared_ptr<lua_State> L, unsigned int idx,
		T *v, Args *... rest)
{
	return Converter<T>::from(L, idx, v) + pop_from_lua(L, idx + 1, rest...);
}
}  // namespace _impl

/**
 *  @class Object
 *  \brief interface to Lua Script
 */
class Object
{
	std::shared_ptr<lua_State> L_;

	int GLOBAL_REF_IDX_;
	int self_;
	std::string path_;

public:

	typedef Object this_type;

	Object() :
			L_(nullptr), self_(0), GLOBAL_REF_IDX_(0)
	{
	}

	Object(std::shared_ptr<lua_State> l, unsigned int G, unsigned int s,
			std::string const &path = "") :
			L_(l), GLOBAL_REF_IDX_(G), self_(s), path_(path)
	{
	}

	Object(Object const &r) :
			L_(r.L_), GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_), path_(r.path_)
	{
		if (L_ != nullptr)
		{
			lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, r.self_);
			self_ = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
		}
	}

	Object(Object &&r) :
			L_(r.L_), GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_), self_(r.self_), path_(
			r.path_)
	{
		r.self_ = 0;
	}

	Object &operator=(Object const &r)
	{
		this->L_ = r.L_;
		this->GLOBAL_REF_IDX_ = r.GLOBAL_REF_IDX_;
		this->path_ = r.path_;
		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, r.self_);
		self_ = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
		return *this;
	}

	void swap(Object &other)
	{
		std::swap(L_, other.L_);
		std::swap(GLOBAL_REF_IDX_, other.GLOBAL_REF_IDX_);
		std::swap(self_, other.self_);
		std::swap(path_, other.path_);

	}

	~Object()
	{

		if (L_ == nullptr)
		{
			return;
		}
		if (self_ > 0)
		{
			luaL_unref(L_.get(), GLOBAL_REF_IDX_, self_);
		}

		if (L_.unique())
		{
			lua_remove(L_.get(), GLOBAL_REF_IDX_);
		}
	}

	inline std::basic_ostream<char> &Serialize(std::basic_ostream<char> &os)
	{
		int top = lua_gettop(L_.get());
		for (int i = 1; i < top; ++i)
		{
			int t = lua_type(L_.get(), i);
			switch (t)
			{
			case LUA_TSTRING:
				os << "[" << i << "]=" << lua_tostring(L_.get(), i)
						<< std::endl;
				break;

			case LUA_TBOOLEAN:
				os << "[" << i << "]=" << std::boolalpha
						<< lua_toboolean(L_.get(), i) << std::endl;
				break;

			case LUA_TNUMBER:
				os << "[" << i << "]=" << lua_tonumber(L_.get(), i)
						<< std::endl;
				break;
			case LUA_TTABLE:
				os << "[" << i << "]=" << "is a table" << std::endl;
				break;
			default:
				os << "[" << i << "]=" << "is an unknown type" << std::endl;
			}
		}
		os << "--  End the listing --" << std::endl;

		return os;
	}

	inline bool is_null() const
	{
		return L_ == nullptr;
	}

	inline bool empty() const // STL style
	{
		return L_ == nullptr;
	}

	operator bool() const
	{
		return L_ != nullptr;
	}

	bool is_global() const
	{
		return L_ != nullptr && self_ == -1;
	}

#define DEF_TYPE_CHECK(_FUN_NAME_, _LUA_FUN_)                              \
    inline bool _FUN_NAME_() const                                        \
    {   bool res=false;                                                   \
        if(L_!=nullptr)                                                   \
        {                                                                 \
          lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);                  \
           res = _LUA_FUN_(L_.get(), -1);                                 \
          lua_pop(L_.get(), 1);                                           \
        }                                                                 \
        return res;                                                       \
    }

	DEF_TYPE_CHECK(is_nil, lua_isnil)

	DEF_TYPE_CHECK(is_number, lua_isnumber)

	DEF_TYPE_CHECK(is_string, lua_isstring)

	DEF_TYPE_CHECK(is_boolean, lua_isboolean)

	DEF_TYPE_CHECK(is_lightuserdata, lua_islightuserdata)

	DEF_TYPE_CHECK(is_function, lua_isfunction)

	DEF_TYPE_CHECK(is_thread, lua_isthread)

	DEF_TYPE_CHECK(is_table, lua_istable)

#undef DEF_TYPE_CHECK

	inline std::string get_typename() const
	{
		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		std::string res = lua_typename(L_.get(), -1);
		lua_pop(L_.get(), 1);
		return res;
	}

	void init()
	{
		if (self_ == 0 || L_ == nullptr)
		{
			L_ = std::shared_ptr<lua_State>(luaL_newstate(), lua_close);

			luaL_openlibs(L_.get());

			lua_newtable(L_.get());  // new table on stack

			GLOBAL_REF_IDX_ = lua_gettop(L_.get());

			self_ = -1;

			path_ = "<GLOBAL>";

		}
	}

	inline void parse_file(std::string const &filename)
	{
		init();
		if (filename != "")
		{
			LUA_ERROR(luaL_dofile(L_.get(), filename.c_str()));
//			LOGGER << "Load Lua file:[" << filename << "]" << std::endl;

		}
	}

	inline void parse_string(std::string const &str)
	{
		init();

		LUA_ERROR(luaL_dostring(L_.get(), str.c_str()))

	}

	class iterator
	{
		std::shared_ptr<lua_State> L_;
		int GLOBAL_IDX_;
		int parent_;
		int key_;
		int value_;
		std::string path_;
	public:
		void Next()
		{
			if (L_ == nullptr)
			{
				return;
			}

			lua_rawgeti(L_.get(), GLOBAL_IDX_, parent_);

			int tidx = lua_gettop(L_.get());

			if (lua_isnil(L_.get(), tidx))
			{
//				LOGIC_ERROR(path_ + " is not iteraterable!");
			}

			if (key_ == LUA_NOREF)
			{
				lua_pushnil(L_.get());
			}
			else
			{
				lua_rawgeti(L_.get(), GLOBAL_IDX_, key_);
			}

			int v, k;

			if (lua_next(L_.get(), tidx))
			{
				v = luaL_ref(L_.get(), GLOBAL_IDX_);
				k = luaL_ref(L_.get(), GLOBAL_IDX_);
			}
			else
			{
				k = LUA_NOREF;
				v = LUA_NOREF;
			}
			if (key_ != LUA_NOREF)
			{
				luaL_unref(L_.get(), GLOBAL_IDX_, key_);
			}
			if (value_ != LUA_NOREF)
			{
				luaL_unref(L_.get(), GLOBAL_IDX_, value_);
			}

			key_ = k;
			value_ = v;

			lua_pop(L_.get(), 1);
		}

	public:
		iterator() :
				L_(nullptr), GLOBAL_IDX_(0), parent_(LUA_NOREF), key_(
				LUA_NOREF), value_(LUA_NOREF)
		{

		}

		iterator(iterator const &r) :
				L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_)
		{
			if (L_ == nullptr)
			{
				return;
			}

			lua_rawgeti(L_.get(), GLOBAL_IDX_, r.parent_);

			parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

			lua_rawgeti(L_.get(), GLOBAL_IDX_, r.key_);

			key_ = luaL_ref(L_.get(), GLOBAL_IDX_);

			lua_rawgeti(L_.get(), GLOBAL_IDX_, r.value_);

			value_ = luaL_ref(L_.get(), GLOBAL_IDX_);

		}

		iterator(iterator &&r) :
				L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), parent_(r.parent_), key_(
				r.key_), value_(r.value_)
		{
			r.parent_ = LUA_NOREF;
			r.key_ = LUA_NOREF;
			r.value_ = LUA_NOREF;
		}

		iterator(std::shared_ptr<lua_State> L, unsigned int G, unsigned int p,
				std::string path) :
				L_(L), GLOBAL_IDX_(G), parent_(p), key_(LUA_NOREF), value_(
				LUA_NOREF), path_(path + "[iterator]")
		{
			if (L_ == nullptr)
			{
				return;
			}

			lua_rawgeti(L_.get(), GLOBAL_IDX_, p);
			bool is_table = lua_istable(L_.get(), -1);
			parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

			if (!is_table)
			{
//				LOGIC_ERROR("Object is not indexable!");
			}
			else
			{
				Next();
			}

		}

		~iterator()
		{
			if (L_ == nullptr)
			{
				return;
			}
			if (key_ != LUA_NOREF)
			{
				luaL_unref(L_.get(), GLOBAL_IDX_, key_);
			}
			if (value_ != LUA_NOREF)
			{
				luaL_unref(L_.get(), GLOBAL_IDX_, value_);
			}
			if (parent_ != LUA_NOREF)
			{
				luaL_unref(L_.get(), GLOBAL_IDX_, parent_);
			}
			if (L_.unique())
			{
				lua_remove(L_.get(), GLOBAL_IDX_);
			}
//			if (L_ != nullptr)
//				CHECK(lua_rawlen(L_.get(), GLOBAL_IDX_));
		}

		bool operator!=(iterator const &r) const
		{
			return (r.key_ != key_);
		}

		std::pair<Object, Object> operator*()
		{
			if (key_ == LUA_NOREF || value_ == LUA_NOREF)
			{
				LOGIC_ERROR("the value of this iterator is invalid!");
			}

			lua_rawgeti(L_.get(), GLOBAL_IDX_, key_);

			int key = luaL_ref(L_.get(), GLOBAL_IDX_);

			lua_rawgeti(L_.get(), GLOBAL_IDX_, value_);

			int value = luaL_ref(L_.get(), GLOBAL_IDX_);

			return std::make_pair(Object(L_, GLOBAL_IDX_, key, path_ + ".key"),
					Object(L_, GLOBAL_IDX_, value, path_ + ".value"));
		}

		iterator &operator++()
		{
			Next();
			return *this;
		}
	};

	iterator begin()
	{
		if (empty())
		{
			return end();
		} else
		{
			return iterator(L_, GLOBAL_REF_IDX_, self_, path_);
		}
	}

	iterator end()
	{
		return iterator();
	}

	iterator begin() const
	{
		return iterator(L_, GLOBAL_REF_IDX_, self_, path_);
	}

	iterator end() const
	{
		return iterator();
	}

	template<typename T> inline Object get_child(T const &key) const
	{
		if (is_null())
		{
			return Object();
		}

		return std::move(at(key));
	}

	size_t size() const
	{
		if (is_null())
		{
			return 0;
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		size_t res = lua_rawlen(L_.get(), -1);
		lua_pop(L_.get(), 1);
		return std::move(res);
	}

	inline Object operator[](char const s[]) const noexcept
	{
		return operator[](std::string(s));
	}

	inline Object operator[](std::string const &s) const noexcept
	{
		if (!(is_table() || is_global()))
		{
			return Object();
		}

		if (is_global())
		{
			lua_getglobal(L_.get(), s.c_str());
		}
		else
		{

			lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
			lua_getfield(L_.get(), -1, s.c_str());
		}

		if (lua_isnil(L_.get(), lua_gettop(L_.get())))
		{
			lua_pop(L_.get(), 1);
			return std::move(Object());
		}
		else
		{

			int id = luaL_ref(L_.get(), GLOBAL_REF_IDX_);

			if (!is_global())
			{
				lua_pop(L_.get(), 1);
			}

			return std::move(Object(L_, GLOBAL_REF_IDX_, id, path_ + "." + s));
		}
	}

	//! unsafe fast access, no boundary check, no path information
	inline Object operator[](int s) const noexcept
	{
		if (!(is_table() || is_global()))
		{
			return Object();
		}

		if (self_ < 0 || L_ == nullptr)
		{
			LOGIC_ERROR(path_ + " is not indexable!");
		}
		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		int tidx = lua_gettop(L_.get());
		lua_rawgeti(L_.get(), tidx, s + 1);
		int res = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
		lua_pop(L_.get(), 1);

		return std::move(Object(L_, GLOBAL_REF_IDX_, res));

	}

	//! index operator with out_of_range exception
	template<typename TIDX> inline Object at(TIDX const &s) const
	{
		if (!(is_table() || is_global()))
		{
			return Object();
		}

		Object res = this->operator[](s);
		if (res.is_null())
		{

			throw (std::out_of_range(
					type_cast<std::string>(s) + "\" is not an element in "
							+ path_));
		}

		return std::move(res);

	}

	//! safe access, with boundary check, no path information
	inline Object at(int s) const
	{
		if (!(is_table() || is_global()))
		{
			return Object();
		}

		if (self_ < 0 || L_ == nullptr)
		{
			LOGIC_ERROR(path_ + " is not indexable!");
		}

//		if (s > size())
//		{
//			throw(std::out_of_range(
//					"index out of range! " + path_ + "[" + type_cast<std::string>(s)
//							+ " > " + type_cast<std::string>(size()) + " ]"));
//		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		int tidx = lua_gettop(L_.get());
		lua_rawgeti(L_.get(), tidx, s + 1);
		int res = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
		lua_pop(L_.get(), 1);

		return std::move(
				Object(L_, GLOBAL_REF_IDX_, res,
						path_ + "[" + type_cast<std::string>(s) + "]"));

	}

	template<typename ...Args>
	Object operator()(Args &&... args) const
	{

		if (is_null())
		{
			WARNING << "Try to call a null Object." << std::endl;
			return Object();
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);

		int idx = lua_gettop(L_.get());

		if (!lua_isfunction(L_.get(), idx))
		{
			return Object(*this);
		}
		else
		{
			LUA_ERROR(lua_pcall(L_.get(),
					_impl::push_to_lua(L_, std::forward<Args>(args)...), 1, 0));

			return Object(L_, GLOBAL_REF_IDX_,
					luaL_ref(L_.get(), GLOBAL_REF_IDX_), path_ + "[ret]");
		}

	}

	template<typename T, typename ...Args> inline T create_object(
			Args &&... args) const
	{
		if (is_null())
		{
			return std::move(T());
		}
		else
		{
			return std::move(T(*this, std::forward<Args>(args)...));
		}

	}

	template<typename T>
	inline T as() const
	{
		T res;
		as(&res);
		return std::move(res);
	}

	template<typename T>
	operator T() const
	{
		return as<T>();
	}

	operator std::basic_string<char>() const
	{
		return as<std::string>();
	}

	template<typename ... T> inline std::tuple<T...> as_tuple() const
	{
		return std::move(as<std::tuple<T...>>());
	}

	template<typename TRect, typename ...Args> void as(
			std::function<TRect(Args ...)> *res) const
	{

		if (is_number() || is_table())
		{
			auto value = this->as<TRect>();

			*res = [value](Args ...args) -> TRect { return value; };

		}
		else if (is_function())
		{
			Object obj = *this;
			*res = [obj](Args ...args) -> TRect { return obj(std::forward<Args>(args)...).template as<TRect>(); };
		}

	}

	template<typename T> inline T as(T const &default_value) const
	{
		T res = default_value;
		as(&res);
		return (res);
	}

	template<typename T> inline bool as(T *res) const
	{
		if (!is_null())
		{
			lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
			_impl::pop_from_lua(L_, lua_gettop(L_.get()), res);
			lua_pop(L_.get(), 1);

			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename T> inline void set(std::string const &name, T const &v)
	{

		if (is_null())
		{
			return;
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		_impl::push_to_lua(L_, v);
		lua_setfield(L_.get(), -2, name.c_str());
		lua_pop(L_.get(), 1);
	}

	template<typename T> inline void set(int s, T const &v)
	{
		if (is_null())
		{
			return;
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		_impl::push_to_lua(L_, v);
		lua_rawseti(L_.get(), -2, s);
		lua_pop(L_.get(), 1);
	}

	template<typename T> inline void add(T const &v)
	{
		if (is_null())
		{
			return;
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		_impl::push_to_lua(L_, v);
		size_t len = lua_rawlen(L_.get(), -1);
		lua_rawseti(L_.get(), -2, len + 1);
		lua_pop(L_.get(), 1);
	}

	/**
	 *
	 * @param name the field name of table ,if name=="" use lua_settable, else append
	 *        new table to the end of parent table
	 * @param narr is a hint for how many elements the table will have as a sequence;
	 * @param nrec is a hint for how many other elements the table will have.
	 * @return a Object of new table
	 *
	 * Lua may use these hints to preallocate memory for the new table.
	 *  This pre-allocation is useful for performance when you know in advance how
	 *   many elements the table will have.
	 *
	 *  \note Lua.org:createtable
	 */
	inline Object new_table(std::string const &name, unsigned int narr = 0,
			unsigned int nrec = 0)
	{
		if (is_null())
		{
			return Object();
		}

		lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
		int tidx = lua_gettop(L_.get());
		lua_createtable(L_.get(), narr, nrec);
		if (name == "")
		{
			int len = lua_rawlen(L_.get(), tidx);
			lua_rawseti(L_.get(), tidx, len + 1);
			lua_rawgeti(L_.get(), tidx, len + 1);
		}
		else
		{
			lua_setfield(L_.get(), tidx, name.c_str());
			lua_getfield(L_.get(), tidx, name.c_str());
		}
		Object res(L_, GLOBAL_REF_IDX_, luaL_ref(L_.get(), GLOBAL_REF_IDX_),
				path_ + "." + name);
		lua_pop(L_.get(), 1);
		return std::move(res);
	}
};

/**
 *
 *     @ingroup Convert
 *     @{
 */
#define DEF_LUA_TRANS(_TYPE_, _TO_FUN_, _FROM_FUN_, _CHECK_FUN_)                                     \
template<> struct Converter<_TYPE_>                                                    \
{                                                                                     \
    typedef _TYPE_ value_type;                                                        \
                                                                                      \
    static inline  unsigned int  from(std::shared_ptr<lua_State>L,  unsigned int  idx, value_type * v,                    \
            value_type const &default_value=value_type())                            \
    {                                                                                 \
        if (_CHECK_FUN_(L.get(), idx))                                                     \
        {                                                                             \
            *v = _FROM_FUN_(L.get(), idx);                                                   \
        }                                                                             \
        else                                                                          \
        {   *v = default_value;                                                       \
        }                                                                             \
        return 1;                                                                     \
    }                                                                                 \
    static inline  unsigned int  to(std::shared_ptr<lua_State>L, value_type const & v)                       \
    {                                                                                 \
        _TO_FUN_(L.get(), v);return 1;                                                \
    }                                                                                 \
};                                                                                    \


DEF_LUA_TRANS(double, lua_pushnumber, lua_tonumber, lua_isnumber)

DEF_LUA_TRANS(int, lua_pushinteger, lua_tointeger, lua_isnumber)

DEF_LUA_TRANS(unsigned
		int, lua_pushunsigned, lua_tounsigned, lua_isnumber)

DEF_LUA_TRANS(long, lua_pushinteger, lua_tointeger, lua_isnumber)

DEF_LUA_TRANS(unsigned
		long, lua_pushunsigned, lua_tounsigned, lua_isnumber)

DEF_LUA_TRANS(bool, lua_pushboolean, lua_toboolean, lua_isboolean)

#undef DEF_LUA_TRANS

template<typename T> struct Converter;

template<> struct Converter<std::string>
{
	typedef std::string value_type;

	static inline unsigned int from(std::shared_ptr<lua_State> L,
			unsigned int idx, value_type *v, value_type const &default_value =
	value_type())
	{
		if (lua_isstring(L.get(), idx))
		{
			*v = lua_tostring(L.get(), idx);
		}
		else
		{
			*v = default_value;
			LOGGER << "Can not convert type "
					<< lua_typename(L.get(), lua_type(L.get(), idx))
					<< " to double !" << logical_error_endl;
		}
		return 1;
	}

	static inline unsigned int to(std::shared_ptr<lua_State> L,
			value_type const &v)
	{
		lua_pushstring(L.get(), v.c_str());
		return 1;
	}
};

/** @} LuaTrans */

inline std::ostream &operator<<(std::ostream &os, Object const &obj)
{
	os << obj.as<std::string>();
	return os;
}
/** @} lua_engine*/
}  // namespace lua

namespace traits
{

template<typename TDest>
struct type_cast<lua::Object, TDest>
{
	static constexpr TDest eval(lua::Object const &v)
	{
		return v.as<TDest>();
	}
};

}  // namespace traits

namespace check
{

template<typename, typename ...> struct is_callable;
template<typename, typename> struct is_indexable;

template<typename ...Args>
struct is_callable<lua::Object, Args ...>
{
	static constexpr bool value = true;
};
template<typename Other>
struct is_indexable<lua::Object, Other>
{
	static constexpr bool value = true;
};

}  // namespace check
} // namespace simpla

#endif  // CORE_UTILITIES_LUA_OBJECT_H_
