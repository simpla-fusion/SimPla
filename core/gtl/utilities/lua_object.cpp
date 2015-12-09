/**
 * @file lua_object.cpp
 * @author salmon
 * @date 2015-12-10.
 */
#include "lua_object.h"

namespace simpla { namespace lua
{


Object::Object() : self_(0), GLOBAL_REF_IDX_(0) { }

Object::Object(LuaState l, unsigned int G, unsigned int s,
               std::string const &path) : L_(l), GLOBAL_REF_IDX_(G), self_(s), path_(path)
{
}

Object::Object(Object const &r) :
        L_(r.L_),
        GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_),
        path_(r.path_)
{
    if (!L_.empty())
    {
        if (r.self_ != 0)
        {
            lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, r.self_);
            self_ = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
        }
        else
        {
            self_ = 0;
        }
    }
}

Object::Object(Object &&r) :
        L_(r.L_),
        GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_),
        self_(r.self_),
        path_(r.path_)
{
    r.self_ = 0;
}


void Object::swap(Object &other)
{
    std::swap(L_, other.L_);
    std::swap(GLOBAL_REF_IDX_, other.GLOBAL_REF_IDX_);
    std::swap(self_, other.self_);
    std::swap(path_, other.path_);

}

Object::~Object()
{
    if (!L_.empty())
    {
        if (self_ > 0)
        {
            luaL_unref(L_.get(), GLOBAL_REF_IDX_, self_);
        }

        if (L_.unique()) { lua_remove(L_.get(), GLOBAL_REF_IDX_); }
    }

}

std::basic_ostream<char> &Object::Serialize(std::basic_ostream<char> &os)
{
    ASSERT(!L_.empty());

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


std::string Object::get_typename() const
{
    lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
    std::string res = lua_typename(L_.get(), -1);
    lua_pop(L_.get(), 1);
    return res;
}

void Object::init()
{
    if (self_ == 0 || L_.empty())
    {
        L_.init();
//            L_ = LuaState(luaL_newstate(), lua_close);
        luaL_openlibs(L_.get());

        lua_newtable(L_.get());  // new table on stack

        GLOBAL_REF_IDX_ = lua_gettop(L_.get());

        self_ = -1;

        path_ = "<GLOBAL>";

    }
}

void Object::parse_file(std::string const &filename)
{
    if (filename != "")
    {
        LUA_ERROR(luaL_dofile(L_.get(), filename.c_str()));
//			LOGGER << "Load Lua file:[" << filename << "]" << std::endl;

    }
}

void Object::parse_string(std::string const &str)
{

    LUA_ERROR(luaL_dostring(L_.get(), str.c_str()))

}


void Object::iterator::Next()
{
    if (L_.empty())
    {
        return;
    }

    lua_rawgeti(L_.get(), GLOBAL_IDX_, parent_);

    int tidx = lua_gettop(L_.get());

    if (lua_isnil(L_.get(), tidx))
    {
//				THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not iteraterable!");
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

Object::iterator::iterator() :
        GLOBAL_IDX_(0), parent_(LUA_NOREF), key_(
        LUA_NOREF), value_(LUA_NOREF)
{

}

Object::iterator::iterator(iterator const &r) :
        L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_)
{
    if (L_.empty()) { return; }

    lua_rawgeti(L_.get(), GLOBAL_IDX_, r.parent_);

    parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

    lua_rawgeti(L_.get(), GLOBAL_IDX_, r.key_);

    key_ = luaL_ref(L_.get(), GLOBAL_IDX_);

    lua_rawgeti(L_.get(), GLOBAL_IDX_, r.value_);

    value_ = luaL_ref(L_.get(), GLOBAL_IDX_);

}

Object::iterator::iterator(iterator &&r) :
        L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), parent_(r.parent_), key_(
        r.key_), value_(r.value_)
{
    r.parent_ = LUA_NOREF;
    r.key_ = LUA_NOREF;
    r.value_ = LUA_NOREF;
}

Object::iterator::iterator(LuaState L, unsigned int G, unsigned int p,
                           std::string path) :
        L_(L), GLOBAL_IDX_(G), parent_(p), key_(LUA_NOREF), value_(
        LUA_NOREF), path_(path + "[iterator]")
{
    if (L_.empty()) { return; }

    lua_rawgeti(L_.get(), GLOBAL_IDX_, p);
    bool is_table = lua_istable(L_.get(), -1);
    parent_ = luaL_ref(L_.get(), GLOBAL_IDX_);

    if (!is_table)
    {
//				THROW_EXCEPTION_LOGIC_ERROR("Object is not indexable!");
    }
    else
    {
        Next();
    }

}

Object::iterator::~iterator()
{
    if (L_.empty())
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


std::pair<Object, Object> Object::iterator::value() const
{
    if (key_ == LUA_NOREF || value_ == LUA_NOREF)
    {
        THROW_EXCEPTION_LOGIC_ERROR("the value of this iterator is invalid!");
    }

    lua_rawgeti(L_.get(), GLOBAL_IDX_, key_);

    int key = luaL_ref(L_.get(), GLOBAL_IDX_);

    lua_rawgeti(L_.get(), GLOBAL_IDX_, value_);

    int value = luaL_ref(L_.get(), GLOBAL_IDX_);

    return std::make_pair(Object(L_, GLOBAL_IDX_, key, path_ + ".key"),
                          Object(L_, GLOBAL_IDX_, value, path_ + ".value"));
}


size_t Object::size() const
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


Object Object::operator[](std::string const &s) const noexcept
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
Object Object::operator[](int s) const noexcept
{
    if (!(is_table() || is_global()))
    {
        return Object();
    }

    if (self_ < 0 || L_.empty())
    {
        THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not indexable!");
    }
    lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
    int tidx = lua_gettop(L_.get());
    lua_rawgeti(L_.get(), tidx, s + 1);
    int res = luaL_ref(L_.get(), GLOBAL_REF_IDX_);
    lua_pop(L_.get(), 1);

    return std::move(Object(L_, GLOBAL_REF_IDX_, res));

}

//! index operator with out_of_range exception
Object Object::at(size_t const &s) const
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
Object Object::at(int s) const
{
    if (!(is_table() || is_global()))
    {
        return Object();
    }

    if (self_ < 0 || L_.empty())
    {
        THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not indexable!");
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
Object Object::new_table(std::string const &name, unsigned int narr,
                         unsigned int nrec)
{
    if (is_null()) { return Object(); }

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


}}