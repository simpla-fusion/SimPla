/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * @file lua_object.h
 *
 * @date 2010-9-22
 * @author salmon
 */

#ifndef TOOLBOX_LUA_OBJECT_H_
#define TOOLBOX_LUA_OBJECT_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "LuaObjectExt.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_cast.h"

extern "C" {

#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

namespace simpla {

/**
 * @ingroup toolbox
 * @addtogroup  lua   Lua engine
 *  @{
 */

#define LUA_ERROR(_CMD_)                                                                                    \
    {                                                                                                       \
        int error = _CMD_;                                                                                  \
        if (error != 0) {                                                                                   \
            RUNTIME_ERROR << std::string("\e[1;32m Lua Error:") << lua_tostring(L_.get(), -1) << "\e[1;37m" \
                          << std::endl;                                                                     \
            lua_pop(L_.get(), 1);                                                                           \
        }                                                                                                   \
    }

inline void try_lua_rawgeti(lua_State *L, int G_INDX_, int key) {
    if (key == -1) {
        lua_pushglobaltable(L);
    } else {
        lua_rawgeti(L, G_INDX_, key);
    }
}
/**
 *  @class LuaObject
 *  \brief interface to Lua Script
 */
class LuaObject {
    struct LuaState {
        struct lua_s {
            lua_State *m_state_;
            std::mutex m_mutex_;

            lua_s() : m_state_(luaL_newstate()) {}
            ~lua_s() { lua_close(m_state_); }

            lua_s(lua_s const &) = delete;
            lua_s(lua_s &&) = delete;
            lua_s &operator=(lua_s const &) = delete;
            lua_s &operator=(lua_s &&) = delete;
        };

        std::shared_ptr<lua_s> m_l_;
        LuaState() : m_l_(nullptr) {}
        explicit LuaState(std::shared_ptr<lua_s> other) : m_l_(std::move(other)) {}
        LuaState(LuaState const &other) = default;
        LuaState(LuaState &&other) noexcept = default;

        ~LuaState() = default;
        LuaState &operator=(LuaState const &other) {
            LuaState(other).swap(*this);
            return *this;
        };
        LuaState &operator=(LuaState &&other) {
            LuaState(other).swap(*this);
            return *this;
        };
        void swap(LuaState &other) { std::swap(m_l_, other.m_l_); };

        void init() { m_l_ = std::make_shared<lua_s>(); }
        bool empty() const { return m_l_ == nullptr; }
        bool unique() const { return m_l_.unique(); }
        struct accessor {
            std::shared_ptr<lua_s> m_l_;
            explicit accessor(std::shared_ptr<lua_s> l) : m_l_(std::move(l)) { m_l_->m_mutex_.lock(); }
            ~accessor() { m_l_->m_mutex_.unlock(); }
            lua_State *operator*() { return m_l_->m_state_; }
            std::shared_ptr<lua_s> get() { return m_l_; }

            accessor(accessor const &) = default;
            accessor(accessor &&) = default;
            accessor &operator=(accessor const &) = default;
            accessor &operator=(accessor &&) = default;
        };

        struct const_accessor {
            std::shared_ptr<lua_s> m_l_;
            explicit const_accessor(std::shared_ptr<lua_s> l) : m_l_(std::move(l)) { m_l_->m_mutex_.lock(); }
            ~const_accessor() { m_l_->m_mutex_.unlock(); }
            lua_State *operator*() { return m_l_->m_state_; }
            std::shared_ptr<lua_s> get() const { return m_l_; }

            const_accessor(const_accessor const &) = default;
            const_accessor(const_accessor &&) = default;
            const_accessor &operator=(const_accessor const &) = default;
            const_accessor &operator=(const_accessor &&) = default;
        };

        accessor acc() { return accessor(m_l_); }
        const_accessor acc() const { return const_accessor(m_l_); }
        bool try_lock() const { return m_l_->m_mutex_.try_lock(); }
        lua_State *get() { return m_l_->m_state_; }
        lua_State *get() const { return const_cast<lua_State *>(m_l_->m_state_); }
    };

   public:
    LuaState L_;

   private:
    int GLOBAL_REF_IDX_;
    int self_;
    std::string path_;

   public:
    typedef LuaObject this_type;

    LuaObject();
    LuaObject(std::shared_ptr<LuaState::lua_s> const &l, int G, int s, std::string const &path = "");
    LuaObject(LuaObject const &other);
    LuaObject(LuaObject &&r) noexcept;
    LuaObject &operator=(LuaObject const &other) {
        LuaObject(other).swap(*this);
        return *this;
    }
    LuaObject &operator=(LuaObject &&other) noexcept {
        LuaObject(std::forward<LuaObject>(other)).swap(*this);
        return *this;
    }

    void swap(LuaObject &other);
    ~LuaObject();
    std::string name() const;
    std::ostream &Print(std::ostream &os, int indent = 0) const;
    bool is_null() const { return L_.empty(); }
    bool empty() const { return L_.empty(); }
    operator bool() const { return !L_.empty(); }
    bool is_global() const { return !L_.empty() && self_ == -1; }
    bool is_nil() const;
    bool is_lightuserdata() const;
    bool is_function() const;
    bool is_thread() const;
    bool is_boolean() const;
    bool is_number() const;
    bool is_string() const;
    bool is_integer() const;
    bool is_floating_point() const;
    /**
     *  table : key-value map
     *  array : table without key
     */
    bool is_table() const;
    bool is_array() const;
    std::string get_typename() const;
    size_type get_shape(size_type *rank, size_type *extents) const;

    void init();
    void parse_file(std::string const &filename, std::string const &status = "");
    void parse_string(std::string const &str);

    class iterator {
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
        iterator(iterator &&r) noexcept;
        iterator(LuaState L, int G, int p, std::string path);
        ~iterator();

        iterator &operator=(iterator const &r) {
            iterator(r).swap(*this);
            return *this;
        }
        iterator &operator=(iterator &&r) noexcept {
            iterator(r).swap(*this);
            return *this;
        }
        void swap(iterator &other) {
            L_.swap(other.L_);
            std::swap(GLOBAL_IDX_, other.GLOBAL_IDX_);
            std::swap(parent_, other.parent_);
            std::swap(key_, other.key_);
            std::swap(value_, other.value_);
            std::swap(path_, other.path_);
        }
        bool operator!=(iterator const &r) const { return (r.key_ != key_); }
        std::pair<LuaObject, LuaObject> value() const;
        std::pair<LuaObject, LuaObject> operator*() const { return value(); };
        std::pair<LuaObject, LuaObject> operator->() const { return value(); };
        iterator &operator++() { return Next(); }
    };

    iterator begin() { return (empty()) ? end() : iterator(L_, GLOBAL_REF_IDX_, self_, path_); }

    iterator end() { return iterator(); }
    iterator begin() const { return iterator(L_, GLOBAL_REF_IDX_, self_, path_); }
    iterator end() const { return iterator(); }

    //    size_t accept(std::function<void(LuaObject const &, LuaObject const &)> const &) const;

    //    int accept(std::function<void(int, LuaObject &)> const &) const;

    template <typename T>
    LuaObject get_child(T const &key) const {
        if (is_null()) { return LuaObject(); }
        return std::move(at(key));
    }

    size_t size() const;
    bool has(std::string const &key) const;

   public:
    LuaObject operator[](char const s[]) const { return operator[](std::string(s)); }
    LuaObject operator[](std::string const &s) const { return get(s); };

    LuaObject get(std::string const &s) const;
    //! unsafe fast access, no boundary check, no path information
    LuaObject get(int s) const;
    LuaObject operator[](int s) const { return get(s); }

    //! index operator with out_of_range exception
    //    LuaObject at(size_t const &s) const;

    //! safe access, with boundary check, no path information
    LuaObject at(int s) const;

    template <typename... Args>
    LuaObject operator()(Args &&... args) const {
        if (is_null()) {
            WARNING << "Try to call a null GeoObject." << std::endl;
            return LuaObject();
        }

        LuaObject res;
        {
            auto acc = L_.acc();
            try_lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
            int idx = lua_gettop(*acc);
            if (!lua_isfunction(*acc, idx)) {
                LuaObject(acc.get(), GLOBAL_REF_IDX_, self_, path_).swap(res);
            } else {
                LUA_ERROR(lua_pcall(*acc, push_to_lua(*acc, std::make_tuple(std::forward<Args>(args)...)), 1, 0));
                LuaObject(acc.get(), GLOBAL_REF_IDX_, luaL_ref(*acc, GLOBAL_REF_IDX_), path_ + "[ret]").swap(res);
            }
        }
        return std::move(res);
    }

    template <typename T>
    T as() const {
        T res;
        if (!as(&res)) { BAD_CAST; }
        return std::move(res);
    }

    template <typename T>
    operator T() const {
        return as<T>();
    }

    operator std::string() const { return as<std::string>(); }

    template <typename... T>
    std::tuple<T...> as_tuple() const {
        return std::move(as<std::tuple<T...>>());
    }

    template <typename TRect, typename... Args>
    void as(std::function<TRect(Args...)> *res) const {
        if (is_number() || is_table()) {
            auto value = this->as<TRect>();
            *res = [value](Args... args) -> TRect { return value; };
        } else if (is_function()) {
            LuaObject f_obj(*this);

            *res = [f_obj](Args... args) -> TRect {
                TRect t;
                auto v_obj = f_obj(std::forward<Args>(args)...);
                if (!v_obj.template as<TRect>(&t)) { RUNTIME_ERROR << ("convert_database_r error!") << std::endl; }
                return std::move(t);
            };
        }
    }

    template <typename T>
    T as(T const &default_value) const {
        T res;
        return as(&res) ? std::move(res) : default_value;
    }

    template <typename T>
    bool as(T *res) const {
        return get(res, nullptr, nullptr) > 0;
    }
    template <typename... Args>
    int get(Args &&... args) const {
        int count = 0;
        if (!is_null()) {
            auto acc = L_.acc();
            try_lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
            count = pop_from_lua(*acc, lua_gettop(*acc), std::forward<Args>(args)...);
            lua_pop(*acc, 1);
        }
        return count > 0;
    }
    template <typename... Args>
    int set(std::string const &name, Args &&... args) {
        int count = 0;
        if (!is_null()) {
            auto acc = L_.acc();
            try_lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
            count = push_to_lua(*acc, std::forward<Args>(args)...);
            lua_setfield(*acc, -2, name.c_str());
            lua_pop(*acc, 1);
        }
        return count;
    }

    template <typename... Args>
    int set(int s, Args &&... args) {
        int count = 0;
        if (!is_null()) {
            auto acc = L_.acc();
            try_lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
            count = push_to_lua(*acc, std::forward<Args>(args)...);
            lua_rawseti(*acc, -2, s + 1);
            lua_pop(*acc, 1);
        }
        return count;
    }

    template <typename... Args>
    int add(Args &&... args) {
        if (is_null()) { return 0; }
        auto acc = L_.acc();
        try_lua_rawgeti(*acc, GLOBAL_REF_IDX_, self_);
        push_to_lua(*acc, std::forward<Args>(args)...);
        size_t len = lua_rawlen(*acc, -1);
        lua_rawseti(*acc, -2, static_cast<int>(len + 1));
        lua_pop(*acc, 1);
        return 1;
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
    LuaObject new_table(std::string const &name, unsigned int narr = 0, unsigned int nrec = 0);
};

std::ostream &operator<<(std::ostream &os, LuaObject const &obj);

}  // namespace simpla
#endif  // TOOLBOX_LUA_OBJECT_H_
