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
#include <mutex>
#include "log.h"
#include "../type_cast.h"
#include "lua_object_ext.h"
//
//extern "C"
//{
//
//#include <lua.h>
//#include <lualib.h>
//#include <lauxlib.h>
//
//#if LUA_VERSION_NUM < 502
//#error  need lua version >502
//#endif
//}

namespace simpla { namespace lua
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
    { \
      std::string msg=MAKE_ERROR_MSG("\e[1;32m" ,"Lua Error:",lua_tostring(L_.get(), -1),  "\e[1;37m",""); \
     lua_pop(L_.get(), 1);                                                     \
     throw(std::runtime_error(msg));                                   \
    }                                                                          \
}

class Object;

/**
 *  @class Object
 *  \brief interface to Lua Script
 */
class Object
{


    struct LuaState
    {
        struct lua_s
        {
            lua_State *m_state_;
            std::mutex m_mutex_;

            lua_s() : m_state_(luaL_newstate()) { }

            ~lua_s() { lua_close(m_state_); }
        };

        std::shared_ptr<lua_s> m_l_;

        LuaState() : m_l_(nullptr) { }

        LuaState(LuaState const &other) : m_l_(other.m_l_) { }

        ~LuaState() { }

        void init() { m_l_ = std::make_shared<lua_s>(); }

        bool empty() const { return m_l_ == nullptr; }

        bool unique() const { return m_l_.unique(); }

        lua_State *get() { return m_l_->m_state_; }

        lua_State *get() const { return const_cast<lua_State *>(m_l_->m_state_); }

        void lock() { m_l_->m_mutex_.lock(); }

        void unlock() { m_l_->m_mutex_.unlock(); }

    };


    LuaState L_;

    int GLOBAL_REF_IDX_;
    int self_;
    std::string path_;

public:

    typedef Object this_type;

    Object();

    Object(LuaState l, unsigned int G, unsigned int s,
           std::string const &path = "");

    Object(Object const &r);

    Object(Object &&r);

    Object &operator=(Object const &other)
    {
        Object(other).swap(*this);

        return *this;
    }

    void swap(Object &other);

    ~Object();

    inline std::basic_ostream<char> &Serialize(std::basic_ostream<char> &os);

    inline bool is_null() const
    {
        return L_.empty();
    }

    inline bool empty() const // STL style
    {
        return L_.empty();
    }

    operator bool() const
    {
        return !L_.empty();
    }

    bool is_global() const
    {
        return !L_.empty() && self_ == -1;
    }

#define DEF_TYPE_CHECK(_FUN_NAME_, _LUA_FUN_)                              \
    inline bool _FUN_NAME_() const                                        \
    {   bool res=false;                                                   \
        if(!L_.empty())                                                   \
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

    inline std::string get_typename() const;

    void init();

    void parse_file(std::string const &filename);

    void parse_string(std::string const &str);

    class iterator
    {
        LuaState L_;
        int GLOBAL_IDX_;
        int parent_;
        int key_;
        int value_;
        std::string path_;
    public:
        iterator &Next();

    public:
        iterator();

        iterator(iterator const &r);

        iterator(iterator &&r);

        iterator(LuaState L, unsigned int G, unsigned int p, std::string path);

        ~iterator();

        bool operator!=(iterator const &r) const { return (r.key_ != key_); }

        std::pair<Object, Object> value() const;

        std::pair<Object, Object> operator*() const { return value(); };

        std::pair<Object, Object> operator->() const { return value(); };


        iterator &operator++() { return Next(); }
    };

    iterator begin()
    {
        if (empty()) { return end(); } else { return iterator(L_, GLOBAL_REF_IDX_, self_, path_); }
    }

    iterator end() { return iterator(); }

    iterator begin() const { return iterator(L_, GLOBAL_REF_IDX_, self_, path_); }

    iterator end() const { return iterator(); }

    template<typename T>
    inline Object get_child(T const &key) const
    {
        if (is_null()) { return Object(); }

        return std::move(at(key));
    }

    size_t size() const;

    inline Object operator[](char const s[]) const noexcept
    {
        return operator[](std::string(s));
    }

    Object operator[](std::string const &s) const noexcept;

    //! unsafe fast access, no boundary check, no path information
    Object operator[](int s) const noexcept;

    //! index operator with out_of_range exception
    Object at(size_t const &s) const;

    //! safe access, with boundary check, no path information
    Object at(int s) const;

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
            LUA_ERROR(lua_pcall(L_.get(), _impl::push_to_lua(L_.get(), std::forward<Args>(args)...), 1, 0));

            return Object(L_, GLOBAL_REF_IDX_, luaL_ref(L_.get(), GLOBAL_REF_IDX_), path_ + "[ret]");
        }

    }

    template<typename T, typename ...Args>
    inline T create_object(Args &&... args) const
    {
        if (is_null()) { return std::move(T()); }
        else { return std::move(T(*this, std::forward<Args>(args)...)); }

    }

    template<typename T>
    inline T as() const
    {
        T res;
        as(&res);
        return std::move(res);
    }

    template<typename T>
    operator T() const { return as<T>(); }

    operator std::basic_string<char>() const { return as<std::string>(); }

    template<typename ... T>
    inline std::tuple<T...> as_tuple() const { return std::move(as<std::tuple<T...>>()); }

    template<typename TRect, typename ...Args>
    void as(std::function<TRect(Args ...)> *res) const
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

    template<typename T>
    inline T as(T const &default_value) const
    {
        T res = default_value;
        as(&res);
        return (res);
    }

    template<typename T>
    inline bool as(T *res) const
    {
        if (!is_null())
        {
            lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
            _impl::pop_from_lua(L_.get(), lua_gettop(L_.get()), res);
            lua_pop(L_.get(), 1);

            return true;
        }
        else
        {
            return false;
        }
    }

    template<typename T>
    inline void set(std::string const &name, T const &v)
    {
        if (is_null()) { return; }

        lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
        _impl::push_to_lua(L_.get(), v);
        lua_setfield(L_.get(), -2, name.c_str());
        lua_pop(L_.get(), 1);
    }

    template<typename T>
    inline void set(int s, T const &v)
    {
        if (is_null()) { return; }

        lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
        _impl::push_to_lua(L_.get(), v);
        lua_rawseti(L_.get(), -2, s);
        lua_pop(L_.get(), 1);
    }

    template<typename T>
    inline void add(T const &v)
    {
        if (is_null()) { return; }

        lua_rawgeti(L_.get(), GLOBAL_REF_IDX_, self_);
        _impl::push_to_lua(L_.get(), v);
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
                            unsigned int nrec = 0);
};


inline std::ostream &operator<<(std::ostream &os, Object const &obj)
{
    os << obj.as<std::string>();
    return os;
}
}

} // namespace simpla
namespace simpla
{
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

template<typename, typename ...>
struct is_callable;
template<typename, typename>
struct is_indexable;

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
}
#endif  // CORE_UTILITIES_LUA_OBJECT_H_
