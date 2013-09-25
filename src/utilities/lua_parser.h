/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: LuaWrap.h 1005 2011-01-27 09:15:38Z salmon $
 * luaWrap.h
 *
 *  Created on: 2010-9-22
 *      Author: salmon
 */

#ifndef INCLUDE_LUA_PARSER_H_
#define INCLUDE_LUA_PARSER_H_
#include "lua.hpp"
#include <map>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "log.h"
#include "refcount.h"
#include "fetl/ntuple.h"

namespace simpla
{

#define LUA_ERROR(_L, _MSG_)  ERROR<< (_MSG_)<<std::string("\n") << lua_tostring(_L, 1) ;   lua_pop(_L, 1);

static void stackDump(lua_State *L)
{
	int top = lua_gettop(L);
	for (int i = 1; i < top; ++i)
	{
		int t = lua_type(L, i);
		switch (t)
		{
		case LUA_TSTRING:
			std::cout << "[" << i << "]" << lua_tostring(L,i) << std::endl;
			break;

		case LUA_TBOOLEAN:
			std::cout << "[" << i << "]" << std::boolalpha
					<< lua_toboolean(L, i) << std::endl;
			break;

		case LUA_TNUMBER:
			std::cout << "[" << i << "]" << lua_tonumber(L, i) << std::endl;
			break;
		case LUA_TTABLE:
			std::cout << "[" << i << "]" << "is a table" << std::endl;
			break;
		default:
			std::cout << "[" << i << "]" << "is an unknown type" << std::endl;
		}
	}
	std::cout << "===== End the listing =====" << std::endl;
}

class LuaObject
{
	std::shared_ptr<lua_State> state_holder_;
	std::string key_; //for reference counting
	lua_State * lstate_;
	int parent_;
	int idx_;
public:

	LuaObject() :
			state_holder_(luaL_newstate(), lua_close), //
			parent_(0), idx_(LUA_GLOBALSINDEX), key_(""), lstate_(
					state_holder_.get())
	{
		luaL_openlibs(lstate_);
	}

	LuaObject(std::shared_ptr<lua_State> p, int parent, std::string pkey) :
			state_holder_(p), key_(pkey), parent_(parent), idx_(0), lstate_(
					p.get())
	{
		lua_getfield(lstate_, parent_, key_.c_str());

		if (lua_isnil(lstate_, -1) || lua_isfunction(lstate_,idx_))
		{
			lua_pop(lstate_, 1);
		}
		else
		{
			idx_ = lua_gettop(lstate_);
		}
	}

	LuaObject(std::shared_ptr<lua_State> p, int idx) :
			state_holder_(p), key_(""), parent_(LUA_GLOBALSINDEX), idx_(idx), lstate_(
					p.get())
	{
	}

	~LuaObject()
	{
		if (idx_ != LUA_GLOBALSINDEX && idx_ != 0)
		{
			lua_remove(lstate_, idx_);
		}
	}

	LuaObject operator[](std::string const & sub_key) const
	{

		if (lua_type(lstate_, idx_) != LUA_TTABLE)
		{
			ERROR << key_ << "is not a table!!";
		}

		return (LuaObject(state_holder_, idx_, sub_key));
	}

public:

	template<typename ... Args>
	LuaObject operator()(Args const & ... args) const
	{
		lua_getfield(lstate_, parent_, key_.c_str());

		if (lua_isnil(lstate_, -1))
		{
			LUA_ERROR(lstate_, "\n Can not find key  [" + key_ + "]! ");
		}
		else if (!lua_isfunction(lstate_, -1))
		{
			ERROR << key_ << " is not a function!!";
		}
		push_arg(args...);
		lua_call(lstate_, sizeof...(args), 1);

		return (LuaObject(state_holder_, lua_gettop(lstate_)));
	}
private:
	template<typename T, typename ... Args>
	inline void push_arg(T const & v, Args const & ... rest) const
	{
		push_arg(v);
		push_arg(rest...);
	}

	inline void push_arg(int const & v) const
	{
		lua_pushinteger(lstate_, v);
	}
	inline void push_arg(double const & v) const
	{
		lua_pushnumber(lstate_, v);
	}
	inline void push_arg(std::string const & v) const
	{
		lua_pushstring(lstate_, v.c_str());
	}

public:
	void ParseFile(std::string const & filename)
	{
		if (filename != "" && luaL_dofile(lstate_, filename.c_str()))
		{
			LUA_ERROR(lstate_, "Can not parse file " + filename + " ! ");
		}
	}
	inline void ParseString(std::string const & str)
	{
		if (luaL_dostring(lstate_, str.c_str()))
		{
			LUA_ERROR(lstate_, "Parsing string error! \n\t" + str);
		}
	}

public:
	template<typename T>
	inline void get(T * value) const
	{
		toValue_(idx_, value);
	}

	template<typename T>
	inline void get(T * value, T const & def) const
	{
		if (idx_ != 0)
		{
			get(value);

		}
		else
		{
			*value = def;
		}

	}
	template<typename T>
	inline T as() const
	{
		T res;
		get(&res);
		return (res);

	}
	template<typename T>
	inline T as(T const &d) const
	{
		T res;
		get(&res, d);
		return (res);

	}

private:

	inline void toValue_(int idx, double *res) const
	{
		*res = lua_tonumber(lstate_, idx);
	}
	inline void toValue_(int idx, int *res) const
	{
		*res = lua_tointeger(lstate_, idx);
	}

	inline void toValue_(int idx, long unsigned int *res) const
	{
		*res = lua_tointeger(lstate_, idx);
	}

	inline void toValue_(int idx, long int *res) const
	{
		*res = lua_tointeger(lstate_, idx);
	}

	inline void toValue_(int idx, bool *res) const
	{
		*res = lua_toboolean(lstate_, idx);
	}
	inline void toValue_(int idx, std::string *res) const
	{
		*res = lua_tostring(lstate_, idx);
	}

	template<typename T1, typename T2>
	inline void toValue_(int idx, std::pair<T1, T2> *res) const
	{
		if (lua_istable(lstate_, idx))
		{
			int top = lua_gettop(lstate_);
			if (idx < 0)
			{
				idx += top + 1;
			}
			lua_rawgeti(lstate_, idx, 1);
			toValue_(-1, res->first);
			lua_pop(lstate_, 1);
			lua_rawgeti(lstate_, idx, 2);
			toValue_(-1, res->second);
			lua_pop(lstate_, 1);
		}
		else
		{
			ERROR << (key_ + " is not a std::pair<T1, T2>");
		}
	}

	template<typename T, int N> inline
	void toValue_(int idx, nTuple<N, T> * res) const
	{
		if (lua_istable(lstate_, idx))
		{
			size_t num = lua_objlen(lstate_, idx);
			for (size_t s = 0; s < N; ++s)
			{
				lua_rawgeti(lstate_, idx, s%num+1);
				toValue_(-1, &((*res)[s]));
				lua_pop(lstate_, 1);
			}

		}
		else
		{
			LUA_ERROR(lstate_, key_ + " is not a table ");
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::vector<T> * array) const
	{
		if (lua_istable(lstate_, idx))
		{
			size_t fnum = lua_objlen(lstate_, idx);

			if (fnum > 0)
			{
				if (array->size() < fnum)
				{
					array->resize(fnum);
				}
				for (size_t s = 0; s < fnum; ++s)
				{
					lua_rawgeti(lstate_, idx, s % fnum + 1);
					toValue_(-1, *array[s]);
					lua_pop(lstate_, 1);
				}
			}
		}
		else
		{
			LUA_ERROR(lstate_, key_ + " is not a std::vector<T>");
		}
	}
	template<typename T>
	inline void toValue_(int idx, std::list<T> * list) const
	{
		if (lua_istable(lstate_, idx))
		{
			size_t fnum = lua_objlen(lstate_, idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(lstate_, idx, s % fnum + 1);
				T tmp;
				toValue_(-1, tmp);
				list->push_back(tmp);
				lua_pop(lstate_, 1);
			}
		}
		else
		{
			LUA_ERROR(lstate_, " std::list<T>");
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::map<std::string, T> *res) const
	{
		if (lua_type(lstate_, idx) == LUA_TTABLE)
		{

			typedef std::string KeyType;
			/* table is in the stack at index 'idx' */
			lua_pushnil(lstate_); /* first key */
			T item;
			KeyType key;
			while (lua_next(lstate_, -2))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				toValue_(-2, &key);
				toValue_(-1, &item);
				(*res)[key] = *item;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(lstate_, 1);
			}

		}
		else
		{
			ERROR << (key_ + " is not a std::map<std::string, ValueType>");
		}
		return;

	}

}
;

//template<typename T>
//	inline void toValue_(int idx, T *res)
//	{
//		switch (lua_type(lstate.get(), idx))
//		{
//		case LUA_TBOOLEAN:
//			*res = lua_toboolean(lstate.get(), idx);
//			break;
//		case LUA_TNUMBER:
//			*res = lua_tonumber(lstate.get(), idx);
//			break;
//		case LUA_TTABLE:
//		{
//			//			typedef typename Reference<T>::KeyType KeyType;
//			//			typedef typename Reference<T>::ValueType ValueType;
//			//
//			//			/* table is in the stack at index 'idx' */
//			//			lua_pushnil(lstate.get()); /* first key */
//			//			ValueType item;
//			//			KeyType key;
//			//			while (lua_next(lstate.get(), -2))
//			//			{
//			//				/* uses 'key' (at index -2) and 'value' (at index -1) */
//			//				toValue_(-1, item);
//			//				toValue_(-2, key);
//			//				Reference<T>::index(res, key) = item;
//			//				/* removes 'value'; keeps 'key' for next iteration */
//			//				lua_pop(lstate.get(), 1);
//			//
//			//			}
//			break;
//		}
//		}
//	}

//	template<typename T>
//	inline void getExprTo(std::string const & expr, T * v)
//	{
//		std::string e = std::string("__evalExpr=") + expr;
//
//		if (luaL_dostring(lstate.get(), e.c_str()))
//		{
//			LUA_ERROR(lstate.get(), e);
//		}
//
//		GetValue2("__evalExpr", v);
//	}
//
//	template<typename T>
//	inline void getExprToArray(std::string const & expr, std::shared_ptr<T> v)
//	{
//		std::string e = std::string("__evalExpr=") + expr;
//
//		if (luaL_dostring(lstate.get(), e.c_str()))
//		{
//			LUA_ERROR(lstate.get(), e);
//		}
//
//		fillArray2("__evalExpr", v);
//	}
//	template<typename T>
//	inline void fillArray(std::string const& key, T & array)
//	{
//		lua_getfield(lstate.get(), LUA_GLOBALSINDEX, key.c_str());
//		int idx = lua_gettop(lstate.get());
//		try
//		{
//			lua_fillArray(idx, array, 0);
//		} catch (std::string const & e)
//		{
//			ERROR << ("Can not parse \"" + key + "\" to " + e + " !");
//		}
//		lua_pop(lstate.get(), 1);
//
//	}
//	template<typename T>
//	inline void fillArray2(std::string const & key, std::shared_ptr<T> array)
//	{
//		fillArray(key, *array);
//	}

}// namespace simpla
#endif  // INCLUDE_LUA_PARSER_H_
