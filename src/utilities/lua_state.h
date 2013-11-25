/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id: LuaWrap.h 1005 2011-01-27 09:15:38Z salmon $
 * luaWrap.h
 *
 *  Created on: 2010-9-22
 *      Author: salmon
 */

#ifndef INCLUDE_LUA_PARSER_H_
#define INCLUDE_LUA_PARSER_H_

#include <lua5.2/lua.hpp>
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
#include <stdlib.h>
#include  <map>
#include "../fetl/ntuple.h"
#include "log.h"
#include "utilities.h"

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

class LuaIterator;
class LuaObject;

template<typename T> struct LuaTrans;

inline void ToLua(lua_State*L)
{
}
template<typename T, typename ... Args>
inline void ToLua(lua_State*L, T const & v, Args const & ... rest)
{
	LuaTrans<T>::To(L, v);
	ToLua(L, rest...);
}

inline void FromLua(lua_State*L, int)
{
}

template<typename T, typename ... Args>
inline void FromLua(lua_State*L, int idx, T & v, Args & ... rest)
{
	LuaTrans<T>::From(L, idx, v);
	FromLua(L, idx + 1, rest...);
}

class LuaObject
{
	std::shared_ptr<lua_State> L_;

	int GLOBAL_IDX_;
	int self_;
	std::string path_;

public:

	LuaObject() :
			L_(nullptr), self_(0), GLOBAL_IDX_(0)

	{
	}
	LuaObject(std::shared_ptr<lua_State> l, int G, int s,
			std::string const & path) :
			L_(l), GLOBAL_IDX_(G), self_(s), path_(path)
	{

	}
	LuaObject(LuaObject const & r) :
			L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), path_(r.path_)
	{
		lua_rawgeti(L_.get(), GLOBAL_IDX_, r.self_);
		self_ = luaL_ref(L_.get(), GLOBAL_IDX_);
	}

	LuaObject(LuaObject && r) :
			L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), self_(r.self_), path_(r.path_)
	{
		r.self_ = 0;
	}

	~LuaObject()
	{

		if (self_ > 0)
		{
			luaL_unref(L_.get(), GLOBAL_IDX_, self_);
		}
//		if (L_ != nullptr)
//		{
//			CHECK(lua_gettop(L_.get()));
//			CHECK(lua_rawlen(L_.get(), GLOBAL_IDX_));
//		}

		if (L_.unique())
		{
			lua_remove(L_.get(), GLOBAL_IDX_);
		}
	}

	void Init()
	{
		if (self_ == 0)
		{
			L_ = std::shared_ptr<lua_State>(luaL_newstate(), lua_close);

			luaL_openlibs(L_.get());

			lua_newtable(L_.get());  // new table on stack

			GLOBAL_IDX_ = lua_gettop(L_.get());

			self_ = -1;

			path_ = "<GLOBAL>";

		}
	}

	inline void ParseFile(std::string const & filename)
	{
		Init();
		if (filename != "" && luaL_dofile(L_.get(), filename.c_str()))
		{
			LUA_ERROR(L_.get(), "Can not parse file " + filename + " ! ");
		}
	}
	inline void ParseString(std::string const & str)
	{
		Init();
		if (luaL_dostring(L_.get(), str.c_str()))
		{
			LUA_ERROR(L_.get(), "Parsing string error! \n\t" + str);
		}
	}

	class iterator
	{
		std::shared_ptr<lua_State> L_;
		int GLOBAL_IDX_;
		int parent_;
		int key_;
		int value_;
		std::string path_;
//		LuaObject key_;
		void Next()
		{
			lua_rawgeti(L_.get(), GLOBAL_IDX_, parent_);
			int tidx = lua_gettop(L_.get());

			CHECK(key_);

			if (key_ < 0)
			{
				lua_pushnil(L_.get());
			}
			else
			{
				lua_rawgeti(L_.get(), GLOBAL_IDX_, key_);
			}

			if (lua_next(L_.get(), tidx))
			{
				value_ = luaL_ref(L_.get(), GLOBAL_IDX_);
				key_ = luaL_ref(L_.get(), GLOBAL_IDX_);
			}
			else
			{
				key_ = -1;
				value_ = -1;
			}
			lua_pop(L_.get(), 1);
		}
	public:
		iterator() :
				L_(nullptr), GLOBAL_IDX_(0), parent_(0), key_(-1), value_(-1)
		{

		}
		iterator(std::shared_ptr<lua_State> L, int G, int p, std::string path) :
				L_(L), GLOBAL_IDX_(G), parent_(p), key_(-1), value_(-1), path_(
						path + "[iterator]")
		{

			lua_rawgeti(L_.get(), GLOBAL_IDX_, parent_);
			bool is_table = lua_istable(L_.get(), -1);
			lua_pop(L_.get(), 1);
			if (!is_table)
			{
				LOGIC_ERROR << "Object is not indexable!";
			}
			else
			{
				Next();
			}

		}

		~iterator()
		{
		}

		bool operator!=(iterator const & r) const
		{
			return (r.key_ != key_);
		}
		std::pair<LuaObject, LuaObject> operator*()
		{
			if (key_ < 0 || value_ < 0)
			{
				LOGIC_ERROR << "the value of this iterator is invalid!";
			}
			return std::make_pair(
					LuaObject(L_, GLOBAL_IDX_, key_, path_ + ".key"),
					LuaObject(L_, GLOBAL_IDX_, value_, path_ + ".value"));
		}

		iterator & operator++()
		{
			Next();
			return *this;
		}
	}
	;

	iterator begin()
	{
		return iterator(L_, GLOBAL_IDX_, self_, path_);
	}
	iterator end()
	{
		return iterator();
	}

	template<typename TFun>
	void ForEach(TFun const &fun)
	{
		lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
		int idx = lua_gettop(L_.get());
		if (lua_type(L_.get(), idx) == LUA_TTABLE)
		{
			/* table is in the stack at index 'idx' */
			lua_pushnil(L_.get()); /* first key */
			while (lua_next(L_.get(), idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */
				int value = luaL_ref(L_.get(), GLOBAL_IDX_);
				lua_pushvalue(L_.get(), lua_gettop(L_.get()));
				int key = luaL_ref(L_.get(), GLOBAL_IDX_);
				fun(LuaObject(L_, GLOBAL_IDX_, key, path_ + "[key]"),
						LuaObject(L_, GLOBAL_IDX_, value, path_ + "[value]"));

			}
		}
		lua_pop(L_.get(), 1);
	}

	template<typename T>
	inline LuaObject GetChild(T const & key) const
	{
		return std::move(at(key));
	}

	inline LuaObject operator[](std::string const & s) const
	{
		LuaObject res;
		bool is_global = (self_ < 0);
		if (is_global)
		{
			lua_getglobal(L_.get(), s.c_str());

		}
		else
		{

			lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
			lua_getfield(L_.get(), -1, s.c_str());
		}

		int id = luaL_ref(L_.get(), GLOBAL_IDX_);

		if (!is_global)
		{
			lua_pop(L_.get(), 1);

		}
		return std::move(
				LuaObject(L_, GLOBAL_IDX_, id, path_ + "." + ToString(s)));
	}

	inline LuaObject at(std::string const & s, bool boundary_check = true) const
	{

		bool is_global = (self_ < 0);
		if (is_global)
		{
			lua_getglobal(L_.get(), s.c_str());
		}
		else
		{

			lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
			lua_getfield(L_.get(), -1, s.c_str());
		}

		if (boundary_check && lua_isnil(L_.get(),lua_gettop(L_.get())))
		{
			lua_pop(L_.get(), 1);
			throw(std::out_of_range(
					ToString(s) + "\" is not an element in " + path_));
		}

		int id = luaL_ref(L_.get(), GLOBAL_IDX_);

		if (!is_global)
		{
			lua_pop(L_.get(), 1);
		}

		return std::move(
				LuaObject(L_, GLOBAL_IDX_, id, path_ + "." + ToString(s)));
	}

	inline LuaObject operator[](int s) const
	{
		if (self_ < 0)
		{
			LOGIC_ERROR << path_ << " is not indexable!";
		}
		lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
		int tidx = lua_gettop(L_.get());
		lua_rawgeti(L_.get(), tidx, s);
		int res = luaL_ref(L_.get(), GLOBAL_IDX_);
		lua_pop(L_.get(), 1);
		return std::move(
				LuaObject(L_, GLOBAL_IDX_, res,
						path_ + "[" + ToString(s) + "]"));
	}

	size_t GetLength() const
	{
		lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
		size_t res = lua_rawlen(L_.get(), -1);
		lua_pop(L_.get(), 1);
		return std::move(res);
	}
	inline LuaObject at(int s) const
	{
		size_t max_length = GetLength();

		if (s > max_length)
		{
			OUT_RANGE_ERROR << path_ << " out of range! " << s << ">"
					<< max_length;
		}

		return std::move(this->operator[](s));
	}

	template<typename ...Args>
	LuaObject operator()(Args const &... args) const
	{
		lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);

		int idx = lua_gettop(L_.get());

		if (!lua_isfunction(L_.get(),idx))
		{
			LOGIC_ERROR << path_ << " is not  a function!";
		}

		ToLua(L_.get(), args...);

		lua_pcall(L_.get(), sizeof...(args), 1, 0);

		return LuaObject(L_, GLOBAL_IDX_, luaL_ref(L_.get(), GLOBAL_IDX_),
				path_ + "[ret]");

	}

	template<typename T>
	inline T Get(std::string const & name, T const & default_value = T()) const
	{
		T res;
		try
		{
			res = at(name).as<T>();

		} catch (...)
		{
			res = default_value;
		}

		return std::move(res);
	}

	template<typename T>
	inline T as() const
	{
		T res;

		lua_rawgeti(L_.get(), GLOBAL_IDX_, self_);
		LuaTrans<T>::From(L_.get(), lua_gettop(L_.get()), res);
		lua_pop(L_.get(), 1);

		return (res);
	}

}
;

#define DEF_LUA_TRANS(_TYPE_,_TO_FUN_,_FROM_FUN_,_CHECK_FUN_)                                     \
template<> struct LuaTrans<_TYPE_>                                                    \
{                                                                                     \
	typedef _TYPE_ value_type;                                                        \
                                                                                      \
	static inline void From(lua_State*L, int idx, value_type & v,value_type const &default_value=value_type())                  \
	{                                                                                 \
		if (_CHECK_FUN_(L, idx))                                                     \
		{                                                                             \
			v = _FROM_FUN_(L, idx);                                                   \
		}                                                                             \
		else                                                                          \
		{   v = default_value;                                                                         \
		}                                                                             \
	}                                                                                 \
	static inline void To(lua_State*L, value_type const & v)                       \
	{                                                                                 \
		_TO_FUN_(L, v);                                                               \
	}                                                                                 \
};                                                                                    \

DEF_LUA_TRANS(double, lua_pushnumber, lua_tonumber, lua_isnumber)
DEF_LUA_TRANS(int, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(unsigned int, lua_pushunsigned, lua_tounsigned, lua_isnumber)
DEF_LUA_TRANS(long, lua_pushinteger, lua_tointeger, lua_isnumber)
DEF_LUA_TRANS(unsigned long, lua_pushunsigned, lua_tounsigned, lua_isnumber)
DEF_LUA_TRANS(bool, lua_pushboolean, lua_toboolean, lua_isboolean)
#undef DEF_LUA_TRANS
//"                           ToString(lua_typename(L, lua_type(L, idx)))                      " to "+  __STRING(_TYPE_) <<"!")
template<> struct LuaTrans<std::string>
{
	typedef std::string value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_isstring(L, idx))
		{
			v = lua_tostring(L, idx);
		}
		else
		{
			v = default_value;
//			LOGIC_ERROR << "Can not convert type "
//					<< lua_typename(L, lua_type(L, idx)) << " to double !";
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		lua_pushstring(L, v.c_str());
	}
};

template<int N, typename T> struct LuaTrans<nTuple<N, T>>
{
	typedef nTuple<N, T> value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_istable(L, idx))
		{
			size_t num = lua_rawlen(L, idx);
			for (size_t s = 0; s < N; ++s)
			{
				lua_rawgeti(L, idx, s % num + 1);
				FromLua(L, -1, (v[s]));
				lua_pop(L, 1);
			}

		}
		else
		{
			v = default_value;
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		LOGIC_ERROR << " UNIMPLEMENTED!!";
	}
};

template<typename T> struct LuaTrans<std::vector<T> >
{
	typedef std::vector<T> value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_istable(L, idx))
		{
			size_t fnum = lua_rawlen(L, idx);

			if (fnum > 0)
			{
				if (v.size() < fnum)
				{
					v.resize(fnum);
				}
				for (size_t s = 0; s < fnum; ++s)
				{
					lua_rawgeti(L, idx, s % fnum + 1);
					FromLua(L, -1, v[s]);
					lua_pop(L, 1);
				}
			}

		}
		else
		{
			v = default_value;
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		LOGIC_ERROR << " UNIMPLEMENTED!!";
	}
};
template<typename T> struct LuaTrans<std::list<T> >
{
	typedef std::list<T> value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_istable(L, idx))
		{
			size_t fnum = lua_rawlen(L, idx);

			for (size_t s = 0; s < fnum; ++s)
			{
				lua_rawgeti(L, idx, s % fnum + 1);
				T tmp;
				FromLua(L, -1, tmp);
				v.push_back(tmp);
				lua_pop(L, 1);
			}

		}
		else
		{
			v = default_value;
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		LOGIC_ERROR << " UNIMPLEMENTED!!";
	}
};

template<typename T1, typename T2> struct LuaTrans<std::map<T1, T2> >
{
	typedef std::map<T1, T2> value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_istable(L, idx))
		{
			lua_pushnil(L); /* first key */

			T1 key;
			T2 value;

			while (lua_next(L, idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				FromLua(L, -2, key);
				FromLua(L, -1, value);
				v[key] = value;
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(L, 1);
			}

		}
		else
		{
			v = default_value;
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		LOGIC_ERROR << " UNIMPLEMENTED!!";
	}
};

template<typename T1, typename T2> struct LuaTrans<std::pair<T1, T2> >
{
	typedef std::pair<T1, T2> value_type;

	static inline void From(lua_State*L, int idx, value_type & v,
			value_type const &default_value = value_type())
	{
		if (lua_istable(L, idx))
		{
			lua_pushnil(L); /* first key */
			while (lua_next(L, idx))
			{
				/* uses 'key' (at index -2) and 'value' (at index -1) */

				FromLua(L, -2, (v.first));
				FromLua(L, -1, (v.second));
				/* removes 'value'; keeps 'key' for next iteration */
				lua_pop(L, 1);
			}

		}
		else
		{
			v = default_value;
		}
	}
	static inline void To(lua_State*L, value_type const & v)
	{
		LOGIC_ERROR << " UNIMPLEMENTED!!";
	}
};

} // namespace simpla

/*
 *
 *
 * struct LuaStateHolder
 {

 std::shared_ptr<LuaStateHolder> parent_;
 std::string key_;
 int self_->idx_;
 lua_State * lstate_;

 LuaStateHolder() :
 idx_(LUA_GLOBALSINDEX), key_(""), lstate_(luaL_newstate())

 {
 luaL_openlibs(lstate_);
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

 LuaStateHolder(std::shared_ptr<LuaStateHolder> p, int idx) :
 parent_(p), idx_(idx), key_(""), lstate_(p->lstate_)
 {
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
 if (key_ != "")
 {
 res = parent_->key_ + "." + key_;
 }
 else
 {
 //				char tmp[20];
 //				itoa(idx_, tmp,10);
 //				res = parent_->key_ + ".[" + tmp + "]";
 }
 }
 return (res);
 }

 };

 class LuaIterator
 {
 std::shared_ptr<LuaStateHolder> holder_;
 public:

 LuaObject operator*()
 {
 return LuaObject(holder_);
 }
 };
 *
 * */
//template<typename T>
//	inline void From(int idx, T *res)
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
//			//				From(-1, item);
//			//				From(-2, key);
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
#endif  // INCLUDE_LUA_PARSER_H_
