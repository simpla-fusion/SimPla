/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: LuaWrap.h 1005 2011-01-27 09:15:38Z salmon $
 * luaWrap.h
 *
 *  Created on: 2010-9-22
 *      Author: salmon
 */

#ifndef INCLUDE_LUA_PARSER_H_
#define INCLUDE_LUA_PARSER_H_

#include <fetl/ntuple.h>
#include <lua5.1/lua.h>
#include <lua5.1/lua.hpp>
#include <utilities/log.h>
#include <cstddef>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

struct LuaStateHolder
{

	std::shared_ptr<LuaStateHolder> parent_;
	std::string key_;
	int idx_;
	lua_State * lstate_;

	LuaStateHolder() :
			idx_(LUA_GLOBALSINDEX), key_(""), lstate_(luaL_newstate())

	{
		luaL_openlibs(lstate_);

		INFORM << "Construct [Root]";
	}
	LuaStateHolder(std::shared_ptr<LuaStateHolder> p, std::string const & key =
			"") :
			parent_(p), idx_(0), key_(key), lstate_(p->lstate_)
	{
		lua_getfield(lstate_, p->idx_, key.c_str());

		idx_ = lua_gettop(lstate_);

		if (lua_isnil(lstate_ , idx_))
		{
			lua_remove(lstate_, idx_);
			idx_ = 0;

			throw std::out_of_range(
					"\"" + key + "\" is not an element in " + p->Path() + "!");
		}
		else if (lua_isfunction(lstate_,idx_))
		{
			lua_remove(lstate_, idx_);
			idx_ = 0;
		}
		else if (!lua_istable(lstate_,idx_))
		{
			lua_remove(lstate_, idx_);
			idx_ = 0;
			throw std::out_of_range(key + " is not a table or function!");
		}
	}

	~LuaStateHolder()
	{

		if (idx_ == LUA_GLOBALSINDEX)
		{
			lua_close(lstate_);
		}
		else if (idx_ != 0)
		{
			lua_remove(lstate_, idx_);
		}

	}

	inline std::string Path() const
	{
		std::string res;
		if (idx_ == LUA_GLOBALSINDEX)
		{
			res = "[root]";
		}
		else
		{
			res = parent_->key_ + "." + key_;
		}
		return (res);
	}

	inline void ParseFile(std::string const & filename)
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
};

class LuaObject
{
	std::shared_ptr<LuaStateHolder> holder_;
public:

	LuaObject() :
			holder_(new LuaStateHolder())
	{
	}

	LuaObject(std::shared_ptr<LuaStateHolder> p, std::string const & key) :
			holder_(new LuaStateHolder(p, key))
	{
	}

	~LuaObject()
	{

	}

	inline void ParseFile(std::string const & filename)
	{
		holder_->ParseFile(filename);
	}
	inline void ParseString(std::string const & str)
	{
		holder_->ParseString(str);
	}

	inline LuaObject GetChild(std::string const & key) const
	{

		return LuaObject(holder_, key);
	}

	template<typename T>
	T Get(std::string const & key, T default_value = T()) const
	{

		T res = default_value;
		try
		{
			GetValue(key, &res);
		} catch (...)
		{
		}

		return (res);
	}

	template<typename T>
	void SetValue(std::string const &name, T const &v)
	{
		push_value(v);
		lua_setfield(holder_->lstate_, holder_->idx_, name.c_str());
	}

	template<typename T>
	void GetValue(std::string const & key, T* res) const
	{
		lua_getfield(holder_->lstate_, holder_->idx_, key.c_str());

		int idx = lua_gettop(holder_->lstate_);

		if (lua_isnil(holder_->lstate_, idx))
		{
			lua_remove(holder_->lstate_, idx);

			throw std::out_of_range(
					"\"" + key + "\" is not an element in " + holder_->Path()
							+ "!");
		}
		else
		{
			toValue_(idx, res);
		}
	}

	template<typename T, typename ... Args>
	T Function(Args const & ... args) const
	{

		T res;
		if (holder_->idx_ != 0)
		{
			ERROR << holder_->Path() << " is not a function!!";
		}

		lua_getfield(holder_->lstate_, holder_->parent_->idx_,
				holder_->key_.c_str());

		push_value(args...);

		lua_call(holder_->lstate_, sizeof...(args), 1);

		toValue_(lua_gettop(holder_->lstate_), &res);

		return res;
	}

private:
	template<typename T, typename ... Args>
	inline void push_value(T const & v, Args const & ... rest) const
	{
		push_value(v);
		push_value(rest...);
	}

	inline void push_value(int const & v) const
	{
		lua_pushinteger(holder_->lstate_, v);
	}
	inline void push_value(double const & v) const
	{
		lua_pushnumber(holder_->lstate_, v);
	}
	inline void push_value(std::string const & v) const
	{
		lua_pushstring(holder_->lstate_, v.c_str());
	}

	inline void toValue_(int idx, double *res) const
	{
		*res = lua_tonumber(holder_->lstate_, idx);
	}
	inline void toValue_(int idx, int *res) const
	{
		*res = lua_tointeger(holder_->lstate_, idx);
	}

	inline void toValue_(int idx, long unsigned int *res) const
	{
		*res = lua_tointeger(holder_->lstate_, idx);
	}

	inline void toValue_(int idx, long int *res) const
	{
		*res = lua_tointeger(holder_->lstate_, idx);
	}

	inline void toValue_(int idx, bool *res) const
	{
		*res = lua_toboolean(holder_->lstate_, idx);
	}
	inline void toValue_(int idx, std::string *res) const
	{
		*res = lua_tostring(holder_->lstate_, idx);
	}

	template<typename T1, typename T2>
	inline void toValue_(int idx, std::pair<T1, T2> *res) const
	{
		if (lua_istable(holder_->lstate_, idx))
		{
			int top = lua_gettop(holder_->lstate_);
			if (idx < 0)
			{
				idx += top + 1;
			}
			lua_rawgeti(holder_->lstate_, idx, 1);
			toValue_(-1, res->first);
			lua_pop(holder_->lstate_, 1);
			lua_rawgeti(holder_->lstate_, idx, 2);
			toValue_(-1, res->second);
			lua_pop(holder_->lstate_, 1);
		}
		else
		{
			ERROR << (holder_->Path() + " is not a std::pair<T1, T2>");
		}
	}
	template<typename T, int N> inline
	void toValue_(int idx, nTuple<N, T> * res) const
	{
		if (lua_istable(holder_->lstate_, idx))
		{
			size_t num = lua_objlen(holder_->lstate_, idx);
			for (size_t s = 0; s < N; ++s)
			{
				lua_rawgeti(holder_->lstate_, idx, s % num + 1);
				toValue_(-1, &((*res)[s]));
				lua_pop(holder_->lstate_, 1);
			}

		}
		else
		{
			LUA_ERROR(holder_->lstate_, holder_->Path() + " is not a table ");
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::vector<T> * array) const
	{
		if (lua_istable(holder_->lstate_, idx))
		{
			size_t fnum = lua_objlen(holder_->lstate_, idx);

			if (fnum > 0)
			{
				if (array->size() < fnum)
				{
					array->resize(fnum);
				}
				for (size_t s = 0; s < fnum; ++s)
				{
					lua_rawgeti(holder_->lstate_, idx, s % fnum + 1);
					toValue_(-1, *array[s]);
					lua_pop(holder_->lstate_, 1);
				}
			}
		}
		else
		{
			LUA_ERROR(holder_->lstate_,
					holder_->Path() + " is not a std::vector<T>");
		}
	}
	template<typename T>
	inline void toValue_(int idx, std::list<T> * list) const
	{
		if (lua_istable(holder_->lstate_, idx))
		{
			size_t fnum = lua_objlen(holder_->lstate_, idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(holder_->lstate_, idx, s % fnum + 1);
				T tmp;
				toValue_(-1, tmp);
				list->push_back(tmp);
				lua_pop(holder_->lstate_, 1);
			}
		}
		else
		{
			LUA_ERROR(holder_->lstate_, " std::list<T>");
		}
	}

	template<typename T>
	inline void toValue_(int idx, std::map<std::string, T> *res) const
	{
		if (lua_type(holder_->lstate_, idx) == LUA_TTABLE)
		{

			typedef std::string KeyType;
			/* table is in the stack at index 'idx' */
			lua_pushnil(holder_->lstate_); /* first key */
			T item;
			KeyType key;
			while (lua_next(holder_->lstate_, -2))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				toValue_(-2, &key);
				toValue_(-1, &item);
				(*res)[key] = *item;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(holder_->lstate_, 1);
			}

		}
		else
		{
			ERROR
					<< (holder_->Path()
							+ " is not a std::map<std::string, ValueType>");
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
