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

//#include "log.h"
#include "fetl/ntuple.h"
namespace simpla
{

#define LUA_ERROR(_L, _MSG_) \
  ERROR<< (_MSG_)<<std::string("\n") << lua_tostring(_L, 1) ; \
  lua_pop(_L, 1); \
  throw -1;

class LuaState
{
	lua_State * lstate;
public:
	typedef LuaState ThisType;
	explicit LuaState() :
			lstate(luaL_newstate())
	{
		luaL_openlibs(lstate);
	}

	~LuaState()
	{
		if (lstate != NULL)
		{
			lua_close(lstate);
			delete lstate;
		}
	}

	void parseFile(std::string const & filename)
	{
		if (filename != "" && luaL_dofile(lstate, filename.c_str()))
		{
			LUA_ERROR(lstate, "Can not parse file "+ filename +" ! ");
		}
	}
	inline void parseString(std::string const & str)
	{
		if (luaL_dostring(lstate, str.c_str()))
		{
			LUA_ERROR(lstate, "Parsing string error! \n\t"+str);
		}
	}

	template<typename T>
	void getValue(std::string const & key, T * value)
	{
		lua_getfield(lstate, LUA_GLOBALSINDEX, key.c_str());
		int idx = lua_gettop(lstate);
		try
		{
			toValue_(idx, value);
		} catch (const char e[])
		{
			ERROR << ("\n\t Can not parse \"" + key + "\" to " + e + "!");
		}
		lua_pop(lstate, 1);
	}
	template<typename T>
	void getValue2(std::string const & key, T* p)
	{
		getValue(key, p);
	}
	template<typename T>
	inline void getExprTo(std::string const & expr, T * v)
	{
		std::string e = std::string("__evalExpr=") + expr;

		if (luaL_dostring(lstate, e.c_str()))
		{
			LUA_ERROR(lstate, e);
		}

		getValue2("__evalExpr", v);
	}

	template<typename T>
	inline void getExprToArray(std::string const & expr, TR1::shared_ptr<T> v)
	{
		std::string e = std::string("__evalExpr=") + expr;

		if (luaL_dostring(lstate, e.c_str()))
		{
			LUA_ERROR(lstate, e);
		}

		fillArray2("__evalExpr", v);
	}
	template<typename T>
	inline void fillArray(std::string const& key, T & array)
	{
		lua_getfield(lstate, LUA_GLOBALSINDEX, key.c_str());
		int idx = lua_gettop(lstate);
		try
		{
			lua_fillArray(idx, array, 0);
		} catch (std::string const & e)
		{
			ERROR << ("Can not parse \"" + key + "\" to " + e + " !");
		}
		lua_pop(lstate, 1);

	}
	template<typename T>
	inline void fillArray2(std::string const & key, TR1::shared_ptr<T> array)
	{
		fillArray(key, *array);
	}

	template<typename T>
	T toValue(std::string const & key)
	{
		T res;
		getValue(key, res);
		return res;
	}

	bool check(std::string const & key) const
	{
		lua_getglobal(lstate, key.c_str());
		bool res = !lua_isnil(lstate, -1);
		lua_pop(lstate, -1);
		return res;
	}
private:
	template<typename T>
	inline void toValue_(int idx, T *res)
	{
		switch (lua_type(lstate, idx))
		{
		case LUA_TBOOLEAN:
			*res = lua_toboolean(lstate, idx);
			break;
		case LUA_TNUMBER:
			*res = lua_tonumber(lstate, idx);
			break;
		case LUA_TTABLE:
		{
//			typedef typename Reference<T>::KeyType KeyType;
//			typedef typename Reference<T>::ValueType ValueType;
//
//			/* table is in the stack at index 'idx' */
//			lua_pushnil(lstate); /* first key */
//			ValueType item;
//			KeyType key;
//			while (lua_next(lstate, -2))
//			{
//				/* uses 'key' (at index -2) and 'value' (at index -1) */
//				toValue_(-1, item);
//				toValue_(-2, key);
//				Reference<T>::index(res, key) = item;
//				/* removes 'value'; keeps 'key' for next iteration */
//				lua_pop(lstate, 1);
//
//			}
			break;
		}
		}
	}
	inline void toValue_(int idx, double &res)
	{
		if (lua_type(lstate, idx) == LUA_TNUMBER)
		{
			res = lua_tonumber(lstate, idx);
		}
		else
		{
			throw(" double");
		}
	}
	inline void toValue_(int idx, int &res)
	{

		if (lua_type(lstate, idx) == LUA_TNUMBER)
		{
			res = lua_tointeger(lstate, idx);
		}
		else
		{
			throw(" int");
		}
	}
	inline void toValue_(int idx, bool &res)
	{

		if (lua_type(lstate, idx) == LUA_TBOOLEAN)
		{
			res = lua_toboolean(lstate, idx);
		}
		else
		{
			throw(" bool");
		}
	}
	inline void toValue_(int idx, std::string &res)
	{
		if (lua_isstring(lstate, idx))
		{
			res = lua_tostring(lstate, idx);
		}
		else
		{
			throw(" std::string");
		}
	}
	template<typename T1, typename T2>
	inline void toValue_(int idx, std::pair<T1, T2> &res)
	{
		if (lua_istable(lstate, idx))
		{
			int top = lua_gettop(lstate);
			if (idx < 0)
			{
				idx += top + 1;
			}
			lua_rawgeti(lstate, idx, 1);
			toValue_(-1, res.first);
			lua_pop(lstate, 1);
			lua_rawgeti(lstate, idx, 2);
			toValue_(-1, res.second);
			lua_pop(lstate, 1);
		}
		else
		{
			throw(" std::pair<T1, T2>");
		}
	}

	template<typename T, int N> inline
	void toValue_(int idx, nTuple<N, T> * res)
	{
		lua_fillArray(idx, res, N);
	}

	template<typename T>
	inline void toValue_(int idx, std::vector<T> * array)
	{
		if (lua_istable(lstate, idx))
		{
			size_t fnum = lua_objlen(lstate, idx);

			if (fnum > 0)
			{
				if (array->size() < fnum)
				{
					array->resize(fnum);
				}
				for (size_t s = 0; s < fnum; ++s)
				{
					lua_rawgeti(lstate, idx, s % fnum + 1);
					toValue_(-1, *array[s]);
					lua_pop(lstate, 1);
				}
			}
		}
		else
		{
			ERROR << (" std::vector<T>");
		}
	}
	template<typename T>
	inline void toValue_(int idx, std::list<T> * list)
	{
		if (lua_istable(lstate, idx))
		{
			size_t fnum = lua_objlen(lstate, idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(lstate, idx, s % fnum + 1);
				T tmp;
				toValue_(-1, tmp);
				list->push_back(tmp);
				lua_pop(lstate, 1);
			}
		}
		else
		{
			ERROR << (" std::list<T>");
		}
	}

	inline void toValue_(int idx, boost::any *res)
	{

		switch (lua_type(lstate, idx))
		{
		case LUA_TBOOLEAN:
			*res = static_cast<bool>(lua_toboolean(lstate, idx));
			break;
		case LUA_TNUMBER:
			*res = static_cast<double>(lua_tonumber(lstate, idx));
			break;
		case LUA_TSTRING:
			*res = std::string(lua_tostring(lstate, idx));
			break;
		default:
			throw(" boost::any");
			break;
		}
	}
	template<typename ValueType>
	inline void toValue_(int idx, std::map<std::string, ValueType> *res)
	{
		if (lua_type(lstate, idx) == LUA_TTABLE)
		{

			typedef std::string KeyType;
			/* table is in the stack at index 'idx' */
			lua_pushnil(lstate); /* first key */
			ValueType item;
			KeyType key;
			while (lua_next(lstate, -2))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				toValue_(-2, key);
				toValue_(-1, item);
				*res[key] = item;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(lstate, 1);
			}

		}
		else
		{
			throw("std::map<std::string, ValueType>");
		}
		return;

	}
	template<typename T>
	inline void lua_fillArray(int idx, T * array, int size)
	{
		if (lua_istable(lstate, idx))
		{
			size_t fnum = lua_objlen(lstate, idx);
			if (fnum > 0)
			{
				for (size_t s = 0; s < size; ++s)
				{
					lua_rawgeti(lstate, idx, s % fnum + 1);
					toValue_(-1, &(*array)[s]);
					lua_pop(lstate, 1);
				}
			}
		}

	}

};
} //namespace simpla
#endif  // INCLUDE_LUA_PARSER_H_
