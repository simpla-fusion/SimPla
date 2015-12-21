/**
 * @file lua_object.cpp
 * @author salmon
 * @date 2015-12-10.
 */
#include "lua_object.h"
#include "lua_object_ext.h"
#include "pretty_stream.h"
#include "../ntuple_ext.h"

namespace simpla { namespace lua
{


LuaObject::LuaObject() : self_(0), GLOBAL_REF_IDX_(0) { }


LuaObject::LuaObject(std::shared_ptr<LuaState::lua_s> const &l, int G, int s, std::string const &path) :
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

LuaObject::LuaObject(LuaObject const &other)
{
    if (!other.empty())
    {
        auto acc = other.L_.acc();
        LuaObject(acc.get(), other.GLOBAL_REF_IDX_, other.self_, other.path_).swap(*this);
    }

}

LuaObject::LuaObject(LuaObject &&r) :
        L_(r.L_),
        GLOBAL_REF_IDX_(r.GLOBAL_REF_IDX_),
        self_(r.self_),
        path_(r.path_)
{
    r.self_ = 0;
}


void LuaObject::swap(LuaObject &other)
{
    std::swap(L_, other.L_);
    std::swap(GLOBAL_REF_IDX_, other.GLOBAL_REF_IDX_);
    std::swap(self_, other.self_);
    std::swap(path_, other.path_);

}

LuaObject::~LuaObject()
{
    if (!L_.empty())
    {
        auto acc = L_.acc();

        if (self_ > 0) { luaL_unref(*acc, GLOBAL_REF_IDX_, self_); }

        if (L_.unique()) { lua_remove(*acc, GLOBAL_REF_IDX_); }
    }

}

std::basic_ostream<char> &LuaObject::Serialize(std::basic_ostream<char> &os)
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


std::string LuaObject::get_typename() const
{
    auto acc = L_.acc();

    lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
    std::string res = lua_typename(*acc, -1);
    lua_pop(*acc, 1);
    return res;
}

void LuaObject::init()
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

void LuaObject::parse_file(std::string const &filename)
{
    if (filename != "")
    {
        auto acc = L_.acc();
        LUA_ERROR(luaL_dofile(*acc, filename.c_str()));
        LOGGER << "Load Lua file:[" << filename << "]" << std::endl;

    }
}

void LuaObject::parse_string(std::string const &str)
{
    auto acc = L_.acc();

    LUA_ERROR(luaL_dostring(*acc, str.c_str()))

}


LuaObject::iterator &LuaObject::iterator::Next()
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

LuaObject::iterator::iterator() :
        GLOBAL_IDX_(0), parent_(LUA_NOREF), key_(LUA_NOREF), value_(LUA_NOREF)
{

}

LuaObject::iterator::iterator(iterator const &r) :
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

LuaObject::iterator::iterator(iterator &&r) :
        L_(r.L_), GLOBAL_IDX_(r.GLOBAL_IDX_), parent_(r.parent_), key_(r.key_), value_(r.value_)
{
    r.parent_ = LUA_NOREF;
    r.key_ = LUA_NOREF;
    r.value_ = LUA_NOREF;
}

LuaObject::iterator::iterator(LuaState L, unsigned int G, unsigned int p, std::string path) :
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

LuaObject::iterator::~iterator()
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


std::pair<LuaObject, LuaObject> LuaObject::iterator::value() const
{
    std::pair<LuaObject, LuaObject> res;

    if (key_ == LUA_NOREF || value_ == LUA_NOREF)
    {
        LOGIC_ERROR << ("the value of this iterator is invalid!") << std::endl;
    }
    else
    {

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_IDX_, key_);

        int key = luaL_ref(*acc, GLOBAL_IDX_);

        lua_rawgeti(*acc, GLOBAL_IDX_, value_);

        int value = luaL_ref(*acc, GLOBAL_IDX_);

        LuaObject(acc.get(), GLOBAL_IDX_, key, path_ + ".key").swap(res.first);

        LuaObject(acc.get(), GLOBAL_IDX_, value, path_ + ".value").swap(res.second);
    }

    return std::move(res);
}


size_t LuaObject::size() const
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


LuaObject LuaObject::operator[](std::string const &s) const noexcept
{
    LuaObject res;

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

            LuaObject(acc.get(), GLOBAL_REF_IDX_, id, path_ + "." + s).swap(res);
        }
    }
    return std::move(res);
}

//! unsafe fast access, no boundary check, no path information
LuaObject LuaObject::operator[](int s) const noexcept
{

    LuaObject r;


    if ((is_table() || is_global()))
    {

        if (self_ < 0 || L_.empty()) { LOGIC_ERROR << (path_ + " is not indexable!") << std::endl; }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        int tidx = lua_gettop(*acc);
        lua_rawgeti(*acc, tidx, s + 1);
        int res = luaL_ref(*acc, GLOBAL_REF_IDX_);
        lua_pop(*acc, 1);
        LuaObject(acc.get(), GLOBAL_REF_IDX_, res).swap(r);

    }
    return std::move(r);

}

//! index operator with out_of_range exception
LuaObject LuaObject::at(size_t const &s) const
{
    LuaObject res;

    if ((is_table() || is_global()))
    {
        LuaObject(this->operator[](s)).swap(res);

        if (res.is_null())
        {
            throw (std::out_of_range(type_cast<std::string>(s) + "\" is not an element in " + path_));
        }
    }
    return std::move(res);

}

//! safe access, with boundary check, no path information
LuaObject LuaObject::at(int s) const
{
    LuaObject r;
    if ((is_table() || is_global()))
    {

        if (self_ < 0 || L_.empty()) { LOGIC_ERROR << (path_ + " is not indexable!"); }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        int tidx = lua_gettop(*acc);
        lua_rawgeti(*acc, tidx, s + 1);
        int res = luaL_ref(*acc, GLOBAL_REF_IDX_);
        lua_pop(*acc, 1);

        LuaObject(acc.get(), GLOBAL_REF_IDX_, res, path_ + "[" + type_cast<std::string>(s) + "]").swap(r);

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
LuaObject LuaObject::new_table(std::string const &name, unsigned int narr, unsigned int nrec)
{
    LuaObject res;
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

        LuaObject(acc.get(), GLOBAL_REF_IDX_, luaL_ref(*acc, GLOBAL_REF_IDX_), path_ + "." + name).swap(res);

        lua_pop(*acc, 1);

    }
    return std::move(res);
}

unsigned int
Converter<Properties>::to(lua_State *&L, Properties const &v)
{
    unsigned int res = -1;

    if (v.size() > 0) { res = Converter<std::map<std::string, Properties>>::to(L, v); }

    else if (v.is_boolean()) { res = Converter<bool>::to(L, v.template as<bool>()); }

    else if (v.is_integral()) { res = Converter<int>::to(L, v.template as<int>()); }

    else if (v.is_floating_point()) { res = Converter<double>::to(L, v.template as<double>()); }

    else if (v.is_string()) { res = Converter<std::string>::to(L, v.template as<std::string>()); }


    return res;

}

unsigned int
Converter<Properties>::from(lua_State *&L, unsigned int idx, Properties *v,
                            Properties const &default_value)
{
    unsigned int success = 0;

    if (lua_istable(L, idx))
    {
        CHECK("lua_istable");

        success = _impl::pop_from_lua(L, idx, dynamic_cast<std::map<std::string, Properties> *>(v));
    }
    else if (lua_isboolean(L, idx))
    {
        CHECK("lua_isboolean");
        bool t;
        success = _impl::pop_from_lua(L, idx, &t);
        *v = t;
    }
    else if (lua_isnumber(L, idx))
    {
        CHECK("lua_isnumber");
        double t;
        success = _impl::pop_from_lua(L, idx, &t);
        *v = t;
    }
    else if (lua_isstring(L, idx))
    {
        CHECK("is_string");

        std::string t;
        success = _impl::pop_from_lua(L, idx, &t);
        *v = t;
    }
    else
    {
        v = default_value;
    }
    return success;
}

bool LuaObject::as(Properties *res) const
{

    bool success = true;
    if (this->is_table())
    {

        bool is_tuple = false;
        if (((*this->begin()).first.is_number())) // ntuple or list
        {

            size_t n = this->size();

            if (this->operator[](0).is_number())
            {

                switch (n)
                {
                    case 1:
                    {
                        double v;
                        success = success && this->as(&v);
                        if (success) (*res) = v;
                        break;
                    }
#define DEF_CASE(_NUM_)  case _NUM_: {  nTuple<double, _NUM_> v; success = success && this->as(&v);  if (success) (*res) = v;}break;

                    DEF_CASE(2)
                    DEF_CASE(3)
                    DEF_CASE(4)
                    DEF_CASE(5)
                    DEF_CASE(6)
                    DEF_CASE(7)
                    DEF_CASE(8)
                    DEF_CASE(9)
                    DEF_CASE(10)
                    default:
                        success = false;
//
                }

#undef DEF_CASE

            }
            else
            {
                switch (n)
                {
                    case 1:
                    {
                        double v;
                        success = success && this->as(&v);
                        if (success) (*res) = v;
                        break;
                    }
#define DEF_CASE(_NUM_)  case _NUM_: {  nTuple<std::string, _NUM_> v; success = success && this->as(&v);  if (success) (*res) = v;break;}
                    DEF_CASE(2)
                    DEF_CASE(3)
                    DEF_CASE(4)
                    DEF_CASE(5)
                    DEF_CASE(6)
                    DEF_CASE(7)
                    DEF_CASE(8)
                    DEF_CASE(9)
                    DEF_CASE(10)
#undef DEF_CASE
                    default:
                        success = false;

                }
            }

        }
        else
        {
            for (auto const &item:*this)
            {
                auto &v = (*res)[item.first.as<std::string>()];
                success = success && item.second.as(&(v));

                if (!success)break;
            }
        }

    }
    else if (this->is_boolean())
    {
        bool v;

        success = success && this->as(&v);

        if (success) (*res) = v;
    }
#if LUA_VERSION_NUM >= 503
        else if (this->lua_isinteger())
        {
            int v;

            success = success && this->as(&v);

            if (success) (*res) = v;
        }
#endif
    else if (this->is_number())
    {
        double v;

        success = success && this->as(&v);

        if (success) (*res) = v;
    }
    else if (this->is_string())
    {
        std::string v;

        success = success && this->as(&v);

        if (success) (*res) = v;
    }
    else
    {
        WARNING << "unknown type can not convert" << std::endl;
//        success = false;
    }

    return success;
}
}}