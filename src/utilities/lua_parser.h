/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: LuaWrap.h 1005 2011-01-27 09:15:38Z salmon $
 * luaWrap.h
 *
 *  Created on: 2010-9-22
 *      Author: salmon
 */

#ifndef INCLUDE_LUA_PARSER_H_
#define INCLUDE_LUA_PARSER_H_
#include <lua5.1/lua.hpp>
#include "include/simpla_defs.h"
#include <map>
#include <list>
#include <vector>
#include <string>
#include <algorithm>

#include "log.h"
#include "fetl/ntuple.h"
namespace simpla
{

#define LUA_ERROR(_L, _MSG_) \
  ERROR<< (_MSG_)<<std::string("\n") << lua_tostring(_L, 1) ; \
  lua_pop(_L, 1); \
  throw -1;

class LuaState
{
	std::string key;
	std::shared_ptr<lua_State> lstate_;

public:

	LuaState() :
			key(""), lstate_(luaL_newstate())
	{
		luaL_openlibs(lstate_.get());
	}

	LuaState(std::string k, std::shared_ptr<lua_State> s) :
			key(k), lstate_(s)
	{

	}

	~LuaState()
	{
		if (lstate_.unique())
		{
			lua_close(lstate_.get());
			lstate_.reset();
		}
	}

	void ParseFile(std::string const & filename)
	{
		if (filename != "" && luaL_dofile(lstate_.get(), filename.c_str()))
		{
			LUA_ERROR(lstate_.get(), "Can not parse file " + filename + " ! ");
		}
	}
	inline void ParseString(std::string const & str)
	{
		if (luaL_dostring(lstate_.get(), str.c_str()))
		{
			LUA_ERROR(lstate_.get(), "Parsing string error! \n\t" + str);
		}
	}

	LuaState operator[](std::string const & sub_key) const
	{
		return (LuaState((key == "" ? sub_key : key + "." + sub_key), lstate_));
	}

	template<typename T>
	inline void get(T & value) const
	{
		lua_getfield(lstate_.get(), LUA_GLOBALSINDEX, key.c_str());

		if (lua_isnil(lstate_.get(), -1))
		{
			ERROR << ("\n Can not find key  [" + key + "]!");
			throw(-1);
		}
		else
		{
			int idx = lua_gettop(lstate_.get());
			toValue_(idx, value);
		}
		lua_pop(lstate_.get(), 1);
	}

	template<typename T>
	inline void get(T & value, T const & def) const
	{
		try
		{
			get(value);

		} catch (...)
		{
			value = def;
		}

	}
	template<typename T>
	inline T as() const
	{
		T res;
		get(res);
		return (res);

	}

	template<typename TV, typename TP>
	inline std::function<TV(TP)> as() const
	{

	}

	bool Check() const
	{
		lua_getglobal(lstate_.get(), key.c_str());
		bool res = !lua_isnil(lstate_.get(), -1);
		lua_pop(lstate_.get(), -1);
		return res;
	}
private:

	inline void toValue_(int idx, double &res) const
	{
		if (lua_type(lstate_.get(), idx) == LUA_TNUMBER)
		{
			res = lua_tonumber(lstate_.get(), idx);
		}
		else
		{
			throw(key + " is not a double!");
		}
	}
	inline void toValue_(int idx, int &res) const
	{

		if (lua_type(lstate_.get(), idx) == LUA_TNUMBER)
		{
			res = lua_tointeger(lstate_.get(), idx);
		}
		else
		{
			throw(key + " is not a int");
		}
	}
	inline void toValue_(int idx, bool &res) const
	{

		if (lua_type(lstate_.get(), idx) == LUA_TBOOLEAN)
		{
			res = lua_toboolean(lstate_.get(), idx);
		}
		else
		{
			throw(key + " is not a bool");
		}
	}
	inline void toValue_(int idx, std::string &res) const
	{
		if (lua_isstring(lstate_.get(), idx))
		{
			res = lua_tostring(lstate_.get(), idx);
		}
		else
		{
			throw(key + " is not a std::string");
		}
	}
	template<typename T1, typename T2>
	inline void toValue_(int idx, std::pair<T1, T2> &res) const
	{
		if (lua_istable(lstate_.get(), idx))
		{
			int top = lua_gettop(lstate_.get());
			if (idx < 0)
			{
				idx += top + 1;
			}
			lua_rawgeti(lstate_.get(), idx, 1);
			toValue_(-1, res.first);
			lua_pop(lstate_.get(), 1);
			lua_rawgeti(lstate_.get(), idx, 2);
			toValue_(-1, res.second);
			lua_pop(lstate_.get(), 1);
		}
		else
		{
			throw(key + " is not a std::pair<T1, T2>");
		}
	}

	template<typename T, int N> inline
	void toValue_(int idx, nTuple<N, T> * res) const
	{
		if (lua_istable(lstate_.get(), idx))
		{
			size_t num = lua_objlen(lstate_.get(), idx);

			for (size_t s = 0, smax = std::min(N, num); s < smax; ++s)
			{
				lua_rawgeti(lstate_.get(), idx, s);
				toValue_(-1, *res[s]);
				lua_pop(lstate_.get(), 1);
			}

		}
		else
		{
			ERROR << (key + " is not a table ");
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::vector<T> * array) const
	{
		if (lua_istable(lstate_.get(), idx))
		{
			size_t fnum = lua_objlen(lstate_.get(), idx);

			if (fnum > 0)
			{
				if (array->size() < fnum)
				{
					array->resize(fnum);
				}
				for (size_t s = 0; s < fnum; ++s)
				{
					lua_rawgeti(lstate_.get(), idx, s % fnum + 1);
					toValue_(-1, *array[s]);
					lua_pop(lstate_.get(), 1);
				}
			}
		}
		else
		{
			ERROR << (key + " is not a std::vector<T>");
		}
	}
	template<typename T>
	inline void toValue_(int idx, std::list<T> * list) const
	{
		if (lua_istable(lstate_.get(), idx))
		{
			size_t fnum = lua_objlen(lstate_.get(), idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(lstate_.get(), idx, s % fnum + 1);
				T tmp;
				toValue_(-1, tmp);
				list->push_back(tmp);
				lua_pop(lstate_.get(), 1);
			}
		}
		else
		{
			ERROR << (" std::list<T>");
		}
	}

	inline void toValue_(int idx, boost::any *res)const
	{

		switch (lua_type(lstate_.get(), idx))
		{
		case LUA_TBOOLEAN:
			*res = static_cast<bool>(lua_toboolean(lstate_.get(), idx));
			break;
		case LUA_TNUMBER:
			*res = static_cast<double>(lua_tonumber(lstate_.get(), idx));
			break;
		case LUA_TSTRING:
			*res = std::string(lua_tostring(lstate_.get(), idx));
			break;
		default:
			throw(" boost::any");
			break;
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::map<std::string, T> *res)const
	{
		if (lua_type(lstate_.get(), idx) == LUA_TTABLE)
		{

			typedef std::string KeyType;
			/* table is in the stack at index 'idx' */
			lua_pushnil(lstate_.get()); /* first key */
			T item;
			KeyType key;
			while (lua_next(lstate_.get(), -2))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				toValue_(-2, key);
				toValue_(-1, item);
				*res[key] = item;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(lstate_.get(), 1);
			}

		}
		else
		{
			throw("std::map<std::string, ValueType>");
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
}//namespace simpla
#endif  // INCLUDE_LUA_PARSER_H_
