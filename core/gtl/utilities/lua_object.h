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

extern "C"
{

#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

#if LUA_VERSION_NUM < 502
#error  need lua version >502
#endif
}

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
      RUNTIME_ERROR<<std::string("\e[1;32m Lua Error:")<<lua_tostring(L_.get(), -1)<< "\e[1;37m"<<std::endl; \
     lua_pop(L_.get(), 1);                                                     \
    }                                                                          \
}


/**
 *  @class GeoObject
 *  \brief interface to Lua Script
 */
class LuaObject
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

        LuaState(std::shared_ptr<lua_s> const &other) : m_l_(other) { }

        LuaState(LuaState const &other) : m_l_(other.m_l_) { }

        ~LuaState() { }

        void init() { m_l_ = std::make_shared<lua_s>(); }

        bool empty() const { return m_l_ == nullptr; }

        bool unique() const { return m_l_.unique(); }

        struct accessor
        {
            std::shared_ptr<lua_s> m_l_;

            accessor(std::shared_ptr<lua_s> const &l) : m_l_(l) { m_l_->m_mutex_.lock(); }

            ~accessor() { m_l_->m_mutex_.unlock(); }

            lua_State *operator*() { return m_l_->m_state_; }

            std::shared_ptr<lua_s> get() { return m_l_; }


        };

        struct const_accessor
        {
            std::shared_ptr<lua_s> m_l_;

            const_accessor(std::shared_ptr<lua_s> const &l) : m_l_(l) { m_l_->m_mutex_.lock(); }

            ~const_accessor() { m_l_->m_mutex_.unlock(); }

            lua_State *operator*() { return m_l_->m_state_; }

            std::shared_ptr<lua_s> get() const { return m_l_; }
        };

        accessor acc() { return accessor(m_l_); }

        const_accessor acc() const { return const_accessor(m_l_); }

        bool try_lock() const { return m_l_->m_mutex_.try_lock(); }

        lua_State *get() { return m_l_->m_state_; }

        lua_State *get() const { return const_cast<lua_State *>(m_l_->m_state_); }


    };


    LuaState L_;

    int GLOBAL_REF_IDX_;
    int self_;
    std::string path_;

public:

    typedef LuaObject this_type;

    LuaObject();

    LuaObject(std::shared_ptr<LuaState::lua_s> const &l, int G, int s, std::string const &path = "");

    LuaObject(LuaObject const &other);

    LuaObject(LuaObject &&r);

    LuaObject &operator=(LuaObject const &other)
    {
        LuaObject(other).swap(*this);

        return *this;
    }

    void swap(LuaObject &other);

    ~LuaObject();

    inline std::basic_ostream<char> &Serialize(std::basic_ostream<char> &os);

    inline bool is_null() const { return L_.empty(); }

    inline bool empty() const { return L_.empty(); }

    operator bool() const { return !L_.empty(); }

    bool is_global() const { return !L_.empty() && self_ == -1; }


    bool is_nil() const;

    bool is_lightuserdata() const;

    bool is_function() const;

    bool is_thread() const;

    bool is_table() const;

    bool is_boolean() const;

    bool is_number() const;

    bool is_string() const;


    bool is_integer()const;
    bool is_floating_point()const;


    /** ntuple < list < table
     *
     *  list =  table without key
     *
     *  ntuple = list with number value
     */
    bool is_list() const;

    bool is_nTuple() const;


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

        std::pair<LuaObject, LuaObject> value() const;

        std::pair<LuaObject, LuaObject> operator*() const { return value(); };

        std::pair<LuaObject, LuaObject> operator->() const { return value(); };


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
    inline LuaObject get_child(T const &key) const
    {
        if (is_null()) { return LuaObject(); }

        return std::move(at(key));
    }

    size_t size() const;

    inline LuaObject operator[](char const s[]) const noexcept
    {
        return operator[](std::string(s));
    }

    LuaObject operator[](std::string const &s) const noexcept;

    //! unsafe fast access, no boundary check, no path information
    LuaObject operator[](int s) const noexcept;

    //! index operator with out_of_range exception
    LuaObject at(size_t const &s) const;

    //! safe access, with boundary check, no path information
    LuaObject at(int s) const;

    template<typename ...Args>
    LuaObject operator()(Args &&... args) const
    {

        if (is_null())
        {
            WARNING << "Try to call a null GeoObject." << std::endl;
            return LuaObject();
        }

        LuaObject res;
        {
            auto acc = L_.acc();

            lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);

            int idx = lua_gettop(*acc);

            if (!lua_isfunction(*acc, idx))
            {
                LuaObject(acc.get(), GLOBAL_REF_IDX_, self_, path_).swap(res);
            }
            else
            {
                LUA_ERROR(lua_pcall(*acc, _impl::push_to_lua(*acc, std::forward<Args>(args)...), 1, 0));

                LuaObject(acc.get(), GLOBAL_REF_IDX_, luaL_ref(*acc, GLOBAL_REF_IDX_), path_ + "[ret]").swap(res);
            }
        }
        return std::move(res);

    }

//        template<typename T, typename ...Args>
//        inline T create_object(Args &&... args) const
//        {
//            if (is_null()) { return std::move(T()); }
//            else { return std::move(T(*this, std::forward<Args>(args)...)); }
//
//        }

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
            LuaObject f_obj(*this);

            *res = [f_obj](Args ...args) -> TRect
            {
                TRect t;

                auto v_obj = f_obj(std::forward<Args>(args)...);

                if (!v_obj.template as<TRect>(&t)) { RUNTIME_ERROR << ("convert error!") << std::endl; }

                return std::move(t);
            };
        }

    }

    template<typename T>
    inline T as(T const &default_value) const
    {
        T res;

        if (!as(&res))
        {
            return default_value;
        }
        else
        {
            return std::move(res);
        };

    }

    template<typename T>
    inline bool as(T *res) const
    {
        if (!is_null())
        {
            auto acc = L_.acc();


            lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);

            auto num = _impl::pop_from_lua(*acc, lua_gettop(*acc), res);

            lua_pop(*acc, 1);

            return num > 0;
        }
        else
        {
            return false;
        }
    }

    bool as(Properties *res) const;

    bool set(std::string const &key, Properties const &res) const;

    template<typename T>
    inline void set(std::string const &name, T const &v)
    {
        if (is_null()) { return; }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);

        _impl::push_to_lua(*acc, v);

        lua_setfield(*acc, -2, name.c_str());

        lua_pop(*acc, 1);


    }

    template<typename T>
    inline void set(int s, T const &v)
    {
        if (is_null()) { return; }

        auto acc = L_.acc();


        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        _impl::push_to_lua(*acc, v);
        lua_rawseti(*acc, -2, s);
        lua_pop(*acc, 1);

    }

    template<typename T>
    inline void add(T const &v)
    {
        if (is_null()) { return; }

        auto acc = L_.acc();

        lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        _impl::push_to_lua(*acc, v);
        size_t len = lua_rawlen(*acc, -1);
        lua_rawseti(*acc, -2, static_cast<int>(len + 1));
        lua_pop(*acc, 1);
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
    inline LuaObject new_table(std::string const &name, unsigned int narr = 0,
                               unsigned int nrec = 0);
};


inline std::ostream &operator<<(std::ostream &os, LuaObject const &obj)
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
struct type_cast<lua::LuaObject, TDest>
{
    static constexpr TDest eval(lua::LuaObject const &v)
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
struct is_callable<lua::LuaObject, Args ...>
{
    static constexpr bool value = true;
};
template<typename Other>
struct is_indexable<lua::LuaObject, Other>
{
    static constexpr bool value = true;
};

}  // namespace check
}
#endif  // CORE_UTILITIES_LUA_OBJECT_H_
