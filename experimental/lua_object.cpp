/*
 * lua_object.cpp
 *
 *  Created on: 2014年12月5日
 *      Author: salmon
 */
#include "lua_object.h"

namespace simpla
{
LuaObject::LuaObject() :
		L_(nullptr), self_(0), GLOBAL_REF_IDX_(0)

{
}

LuaObject::LuaObject(std::shared_ptr<lua_State> l, unsigned int G,
		unsigned int s, std::string const & path) :
		L_(l), GLOBAL_REF_IDX_(G), self_(s), path_(path)
{
}
LuaObject::LuaObject(LuaObject const & r) :
		L_(r.L_), GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_), path_(r.path_)
{

	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, r.self_);
	self_ = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
}

LuaObject::LuaObject(LuaObject && r) :
		L_(r.L_), GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_), self_(r.self_), path_(
				r.path_)
{
	r.self_ = 0;
}

LuaObject & LuaObject::operator=(LuaObject const & r)
{
	this->L_ = r.L_;
	this->GLOBAL_REF_IDX_ = r.GLOBAL_REF_IDX_;
	this->path_ = r.path_;
	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, r.self_);
	self_ = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
	return *this;
}

LuaObject::~LuaObject()
{

	if (self_ > 0)
	{
		luaL_unref(L_.get(), GLOBAL_REF_IDX_, self_);
	}

	if (L_.unique())
	{
		lua_remove(L_.get(), GLOBAL_REF_IDX_);
	}
}

std::basic_ostream<char> & LuaObject::print(std::basic_ostream<char> &os)
{
	int top = lua_gettop(L_.get());
	for (int i = 1; i < top; ++i)
	{
		int t = lua_type(L_.get(), i);
		switch (t)
		{
		case LUA_TSTRING:
			os << "[" << i << "]=" << lua_tostring(L_.get(), i) << std::endl;
			break;

		case LUA_TBOOLEAN:
			os << "[" << i << "]=" << std::boolalpha
					<< lua_toboolean(L_.get(), i) << std::endl;
			break;

		case LUA_TNUMBER:
			os << "[" << i << "]=" << lua_tonumber(L_.get(), i) << std::endl;
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

std::string LuaObject::get_typename() const
{
	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
	std::string res = lua_typename(L_.get(), -1);
	lua_pop(L_.get(), 1);
	return res;
}

void LuaObject::init()
{
	if (self_ == 0)
	{
		L_ = std::shared_ptr<lua_State>(luaL_newstate(), lua_close);

		luaL_openlibs(L_.get());

		lua_newtable(L_.get());  // new table on stack

		GLOBAL_REF_IDX_ = lua_gettop(L_.get());

		self_ = -1;

		path_ = "<GLOBAL>";

	}
}

void LuaObject::parse_file(std::string const & filename)
{
	init();
	if (filename != "" && luaL_dofile(L_.get(), filename.c_str()))
	{
		LUA_ERROR(L_.get(), "Can not parse file " + filename + " ! ");
	}
}
void LuaObject::parse_string(std::string const & str)
{
	init();
	if (luaL_dostring(L_.get(), str.c_str()))
	{
		LUA_ERROR(L_.get(), "Parsing string error! \n\t" + str);
	}
}
size_t LuaObject::size() const
{
	if (IsNull())
		return 0;

	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
	size_t res = lua_rawlen(L_.get(), -1);
	lua_pop(L_.get(), 1);
	return std::move(res);
}

LuaObject LuaObject::operator[](char const s[]) const noexcept
{
	return operator[](std::string(s));
}
LuaObject LuaObject::operator[](std::string const & s) const noexcept
{
	if (IsNull())
		return LuaObject();

	bool is_global = (self_ < 0);
	if (is_global)
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
		return std::move(LuaObject());
	}
	else
	{

		int id = luaL_ref(L_.get(), GLOBAL_REF_IDX_);

		if (!is_global)
		{
			lua_pop(L_.get(), 1);
		}

		return (LuaObject(L_, GLOBAL_REF_IDX_, id, path_ + "." + ToString(s)));
	}
}

//! unsafe fast access, no boundary check, no path information
LuaObject LuaObject::operator[](int s) const noexcept
{
	if (IsNull())
		return LuaObject();

	if (self_ < 0 || L_ == nullptr)
	{
		LOGIC_ERROR(path_ + " is not indexable!");
	}
	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
	int tidx = lua_gettop(L_.get());
	lua_rawgeti(L_.get(), tidx, s + 1);
	int res = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
	lua_pop(L_.get(), 1);

	return std::move(LuaObject(L_, GLOBAL_REF_IDX_, res));

}

//! safe access, with boundary check, no path information
LuaObject LuaObject::at(int s) const
{
	if (IsNull())
		return LuaObject();

	if (self_ < 0 || L_ == nullptr)
	{
		LOGIC_ERROR(path_ + " is not indexable!");
	}

	if (s > size())
	{
		throw(std::out_of_range(
				"index out of range! " + path_ + "[" + ToString(s) + " > "
						+ ToString(size()) + " ]"));
	}

	lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
	int tidx = lua_gettop(L_.get());
	lua_rawgeti(L_.get(), tidx, s + 1);
	int res = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
	lua_pop(L_.get(), 1);

	return std::move(
			LuaObject(L_, GLOBAL_REF_IDX_, res,
					path_ + "[" + ToString(s) + "]"));

}
inline LuaObject LuaObject::new_table(std::string const & name,
		unsigned int narr, unsigned int nrec)
{
	if (IsNull())
		return LuaObject();

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
	LuaObject res(L_, GLOBAL_REF_IDX_, luaL_ref(L_.get(), GLOBAL_REF_IDX_),
			path_ + "." + name);
	lua_pop(L_.get(), 1);
	return std::move(res);
}

void LuaObject::iterator::Next()
{
	lua_rawgeti(L_.get(), GLOBAL_IDX_, parent_);

	int tidx = lua_gettop(L_.get());

	if (lua_isnil(L_.get(), tidx))
	{
		LOGIC_ERROR(path_ + " is not iteraterable!");
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
		luaL_unref(L_.get(), GLOBAL_IDX_, key_);
	if (value_ != LUA_NOREF)
		luaL_unref(L_.get(), GLOBAL_IDX_, value_);

	key_ = k;
	value_ = v;

	lua_pop(L_.get(), 1);
}

LuaObject::iterator::iterator() :
		L_(nullptr), GLOBAL_IDX_(0), parent_(LUA_NOREF), key_(
		LUA_NOREF), value_(LUA_NOREF)
{

}
LuaObject::iterator::iterator(iterator const& r) :
		L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_)
{

	lua_rawgeti(L_.get(), GLOBAL_IDX_, r.parent_);

	parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

	lua_rawgeti(L_.get(), GLOBAL_IDX_, r.key_);

	key_ = luaL_ref(L_.get(), GLOBAL_IDX_);

	lua_rawgeti(L_.get(), GLOBAL_IDX_, r.value_);

	value_ = luaL_ref(L_.get(), GLOBAL_IDX_);

}
LuaObject::iterator::iterator(iterator && r) :
		L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), parent_(r.parent_), key_(r.key_), value_(
				r.value_)
{
	r.parent_ = LUA_NOREF;
	r.key_ = LUA_NOREF;
	r.value_ = LUA_NOREF;
}
LuaObject::iterator::iterator(std::shared_ptr<lua_State> L, unsigned int G,
		unsigned int p, std::string path) :
		L_(L), GLOBAL_IDX_(G), parent_(p), key_(LUA_NOREF), value_(
		LUA_NOREF), path_(path + "[iterator]")
{
	lua_rawgeti(L_.get(), GLOBAL_IDX_, p);
	bool is_table = lua_istable(L_.get(), -1);
	parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

	if (!is_table)
	{
		LOGIC_ERROR("Object is not indexable!");
	}
	else
	{
		Next();
	}

}

LuaObject::iterator::~iterator()
{
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

bool LuaObject::iterator::operator!=(iterator const & r) const
{
	return (r.key_ != key_);
}
std::pair<LuaObject, LuaObject> LuaObject::iterator::operator*()
{
	if (key_ == LUA_NOREF || value_ == LUA_NOREF)
	{
		LOGIC_ERROR("the value of this iterator is invalid!");
	}

	lua_rawgeti(L_.get(), GLOBAL_IDX_, key_);

	int key = luaL_ref(L_.get(), GLOBAL_IDX_);

	lua_rawgeti(L_.get(), GLOBAL_IDX_, value_);

	int value = luaL_ref(L_.get(), GLOBAL_IDX_);

	return std::make_pair(LuaObject(L_, GLOBAL_IDX_, key, path_ + ".key"),
			LuaObject(L_, GLOBAL_IDX_, value, path_ + ".value"));
}

LuaObject::iterator & LuaObject::iterator::operator++()
{
	Next();
	return *this;
}

}  // namespace simpla
