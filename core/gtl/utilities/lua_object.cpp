/**
 * @file lua_object.cpp
 * @author salmon
 * @date 2015-12-10.
 */
#include "lua_object.h"

namespace simpla { namespace lua
{


Object::Object() : self_(0), GLOBAL_REF_IDX_(0) { }


Object::Object(std::shared_ptr<LuaState::lua_s> const &l, int G, int s, std::string const &path) :
        L_(l), GLOBAL_REF_IDX_(G), path_(path)
{
    if (s != 0)
    {
        lua_rawgeti(l->m_state_, GLOBAL_REF_IDX_, s);
        self_ = luaL_ref(l->m_state_, GLOBAL_REF_IDX_);
    }
    else
    {
        self_ = 0;
    }
}

Object::Object(Object const &other)
{
    if (!other.empty())
    {
        auto acc = other.L_.acc();
        Object(acc.get(), other.GLOBAL_REF_IDX_, other.self_, other.path_).swap(*this);
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
        auto acc = L_.acc();

        if (self_ > 0) { luaL_unref(*acc, GLOBAL_REF_IDX_, self_); }

        if (L_.unique()) { lua_remove(*acc, GLOBAL_REF_IDX_); }
    }

}

std::basic_ostream<char> &Object::Serialize(std::basic_ostream<char> &os)
{

    auto acc = L_.acc();

    int top = lua_gettop(*acc);
    for (int i = 1; i < top; ++i)
    {
        int t = lua_type(*acc, i);
        switch (t)
        {
            case LUA_TSTRING:
                os << "[" << i << "]=" << lua_tostring(*acc, i)
                << std::endl;
                break;

            case LUA_TBOOLEAN:
                os << "[" << i << "]=" << std::boolalpha
                << lua_toboolean(*acc, i) << std::endl;
                break;

            case LUA_TNUMBER:
                os << "[" << i << "]=" << lua_tonumber(*acc, i)
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
    auto acc = L_.acc();

    lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
    std::string res = lua_typename(*acc, -1);
    lua_pop(*acc, 1);
    return res;
}

void Object::init()
{
    if (self_ == 0 || L_.empty())
    {
        L_.init();
//            L_ = LuaState(luaL_newstate(), lua_close);
        auto acc = L_.acc();
        luaL_openlibs(*acc);

        lua_newtable(*acc);  // new table on stack

        GLOBAL_REF_IDX_ = lua_gettop(*acc);

        self_ = -1;

        path_ = "<GLOBAL>";

    }
}

void Object::parse_file(std::string const &filename)
{
    if (filename != "")
    {
        auto acc = L_.acc();
        LUA_ERROR(luaL_dofile(*acc, filename.c_str()));
        LOGGER << "Load Lua file:[" << filename << "]" << std::endl;

    }
}

void Object::parse_string(std::string const &str)
{
    auto acc = L_.acc();

    LUA_ERROR(luaL_dostring(*acc, str.c_str()))

}


Object::iterator &Object::iterator::Next()
{
    if (L_.empty()) { return *this; }
    else
    {
        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_IDX_, parent_);

        int tidx = lua_gettop(*acc);

        if (lua_isnil(*acc, tidx))
        {
            THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not iteraterable!");
        }

        if (key_ == LUA_NOREF)
        {
            lua_pushnil(*acc);
        }
        else
        {
            lua_rawgeti(*acc, GLOBAL_IDX_, key_);
        }

        int v, k;

        if (lua_next(*acc, tidx))
        {
            v = luaL_ref(*acc, GLOBAL_IDX_);
            k = luaL_ref(*acc, GLOBAL_IDX_);
        }
        else
        {
            k = LUA_NOREF;
            v = LUA_NOREF;
        }
        if (key_ != LUA_NOREF)
        {
            luaL_unref(*acc, GLOBAL_IDX_, key_);
        }
        if (value_ != LUA_NOREF)
        {
            luaL_unref(*acc, GLOBAL_IDX_, value_);
        }

        key_ = k;
        value_ = v;

        lua_pop(*acc, 1);
    }
    return *this;
}

Object::iterator::iterator() :
        GLOBAL_IDX_(0), parent_(LUA_NOREF), key_(LUA_NOREF), value_(LUA_NOREF)
{

}

Object::iterator::iterator(iterator const &r) :
        L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_)
{
    if (L_.empty()) { return; }
    else
    {
        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_IDX_, r.parent_);

        parent_ = luaL_ref(*acc, GLOBAL_IDX_);

        lua_rawgeti(*acc, GLOBAL_IDX_, r.key_);

        key_ = luaL_ref(*acc, GLOBAL_IDX_);

        lua_rawgeti(*acc, GLOBAL_IDX_, r.value_);

        value_ = luaL_ref(*acc, GLOBAL_IDX_);
    }
}

Object::iterator::iterator(iterator &&r) :
        L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), parent_(r.parent_), key_(r.key_), value_(r.value_)
{
    r.parent_ = LUA_NOREF;
    r.key_ = LUA_NOREF;
    r.value_ = LUA_NOREF;
}

Object::iterator::iterator(LuaState L, unsigned int G, unsigned int p, std::string path) :
        L_(L), GLOBAL_IDX_(G), parent_(p), key_(LUA_NOREF), value_(LUA_NOREF), path_(path + "[iterator]")
{
    if (L_.empty()) { return; }
    else
    {
        auto acc = L_.acc();
        lua_rawgeti(*acc, GLOBAL_IDX_, p);
        bool is_table = lua_istable(*acc, -1);
        parent_ = luaL_ref(*acc, GLOBAL_IDX_);
    }

//    if (!is_table())
//    {
//        THROW_EXCEPTION_LOGIC_ERROR("GeoObject is not indexable!");
//    }
//    else
    {
        Next();
    }

}

Object::iterator::~iterator()
{
    if (L_.empty()) { return; }
    else
    {
        auto acc = L_.acc();

        if (key_ != LUA_NOREF)
        {
            luaL_unref(*acc, GLOBAL_IDX_, key_);
        }
        if (value_ != LUA_NOREF)
        {
            luaL_unref(*acc, GLOBAL_IDX_, value_);
        }
        if (parent_ != LUA_NOREF)
        {
            luaL_unref(*acc, GLOBAL_IDX_, parent_);
        }
        if (L_.unique())
        {
            lua_remove(*acc, GLOBAL_IDX_);
        }
    }
}


std::pair<Object, Object> Object::iterator::value() const
{
    std::pair<Object, Object> res;

    if (key_ == LUA_NOREF || value_ == LUA_NOREF)
    {
        THROW_EXCEPTION_LOGIC_ERROR("the value of this iterator is invalid!");
    }
    else
    {

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_IDX_, key_);

        int key = luaL_ref(*acc, GLOBAL_IDX_);

        lua_rawgeti(*acc, GLOBAL_IDX_, value_);

        int value = luaL_ref(*acc, GLOBAL_IDX_);

        Object(acc.get(), GLOBAL_IDX_, key, path_ + ".key").swap(res.first);

        Object(acc.get(), GLOBAL_IDX_, value, path_ + ".value").swap(res.second);
    }

    return std::move(res);
}


size_t Object::size() const
{
    size_t res = 0;

    if (!is_null())
    {
        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);

        res = lua_rawlen(*acc, -1);

        lua_pop(*acc, 1);
    }
    return res;
}


Object Object::operator[](std::string const &s) const noexcept
{
    Object res;

    if ((is_table() || is_global()))
    {
        auto acc = L_.acc();


        if (is_global()) { lua_getglobal(*acc, s.c_str()); }
        else
        {
            lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
            lua_getfield(*acc, -1, s.c_str());
        }

        if (lua_isnil(*acc, lua_gettop(*acc))) { lua_pop(*acc, 1); }
        else
        {

            int id = luaL_ref(*acc, GLOBAL_REF_IDX_);

            if (!is_global()) { lua_pop(*acc, 1); }

            Object(acc.get(), GLOBAL_REF_IDX_, id, path_ + "." + s).swap(res);
        }
    }
    return std::move(res);
}

//! unsafe fast access, no boundary check, no path information
Object Object::operator[](int s) const noexcept
{

    Object r;


    if ((is_table() || is_global()))
    {

        if (self_ < 0 || L_.empty()) { THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not indexable!"); }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        int tidx = lua_gettop(*acc);
        lua_rawgeti(*acc, tidx, s + 1);
        int res = luaL_ref(*acc, GLOBAL_REF_IDX_);
        lua_pop(*acc, 1);
        Object(acc.get(), GLOBAL_REF_IDX_, res).swap(r);

    }
    return std::move(r);

}

//! index operator with out_of_range exception
Object Object::at(size_t const &s) const
{
    Object res;

    if ((is_table() || is_global()))
    {
        Object(this->operator[](s)).swap(res);

        if (res.is_null())
        {
            throw (std::out_of_range(type_cast<std::string>(s) + "\" is not an element in " + path_));
        }
    }
    return std::move(res);

}

//! safe access, with boundary check, no path information
Object Object::at(int s) const
{
    Object r;
    if ((is_table() || is_global()))
    {

        if (self_ < 0 || L_.empty()) { THROW_EXCEPTION_LOGIC_ERROR(path_ + " is not indexable!"); }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        int tidx = lua_gettop(*acc);
        lua_rawgeti(*acc, tidx, s + 1);
        int res = luaL_ref(*acc, GLOBAL_REF_IDX_);
        lua_pop(*acc, 1);

        Object(acc.get(), GLOBAL_REF_IDX_, res, path_ + "[" + type_cast<std::string>(s) + "]").swap(r);

    }
    return std::move(r);

}


/**
 *
 * @param name the field name of table ,if name=="" use lua_settable, else append
 *        new table to the end of parent table
 * @param narr is a hint for how many elements the table will have as a sequence;
 * @param nrec is a hint for how many other elements the table will have.
 * @return a GeoObject of new table
 *
 * Lua may use these hints to preallocate memory for the new table.
 *  This pre-allocation is useful for performance when you know in advance how
 *   many elements the table will have.
 *
 *  \note Lua.org:createtable
 */
Object Object::new_table(std::string const &name, unsigned int narr, unsigned int nrec)
{
    Object res;
    if (!is_null())
    {
        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);

        int tidx = lua_gettop(*acc);

        lua_createtable(*acc, narr, nrec);

        if (name == "")
        {
            int len = static_cast<int>(lua_rawlen(*acc, tidx));
            lua_rawseti(*acc, tidx, len + 1);
            lua_rawgeti(*acc, tidx, len + 1);
        }
        else
        {
            lua_setfield(*acc, tidx, name.c_str());
            lua_getfield(*acc, tidx, name.c_str());
        }

        Object(acc.get(), GLOBAL_REF_IDX_, luaL_ref(*acc, GLOBAL_REF_IDX_), path_ + "." + name).swap(res);

        lua_pop(*acc, 1);

    }
    return std::move(res);
}


}}