/**
 * @file  config_parser.h
 *
 *  Created on: 2014-11-24
 *      Author: salmon
 */

#ifndef CORE_UTILITIES_CONFIG_PARSER_H_
#define CORE_UTILITIES_CONFIG_PARSER_H_

#include <algorithm>
#include <cstdbool>
#include <map>
#include <string>
#include <utility>

#include "../type_cast.h"
#include "lua_object.h"

namespace simpla
{

/**
 * @ingroup utilities
 */
struct ConfigParser
{
    int argc;
    char **argv;

    ConfigParser();

    ~ConfigParser();

    std::string init(int argc, char **argv);

    struct DictObject : public lua::Object
    {
        DictObject()
                : lua::Object(), m_value_("")
        {

        }

        DictObject(DictObject const &other)
                : lua::Object(other), m_value_(other.m_value_)
        {

        }

        DictObject(DictObject &&other)
                : lua::Object(other), m_value_(other.m_value_)
        {

        }

        DictObject(lua::Object const &lua_obj)
                : lua::Object(lua_obj), m_value_("")
        {

        }

        DictObject(std::string const &value)
                : /*lua::Object(),*/
                m_value_(value)
        {
        }

        ~DictObject()
        {
        }

        void swap(DictObject &other)
        {
            lua::Object::swap(other);
            std::swap(m_value_, other.m_value_);
        }

        DictObject &operator=(DictObject const &other)
        {
            DictObject(other).swap(*this);
            return *this;
        }

        operator bool() const
        {
            return m_value_ != "" || lua::Object::operator bool();
        }

        template<typename T>
        T as() const
        {
            if (m_value_ != "")
            {
                return std::move(type_cast<T>(m_value_));
            }
            else if (!lua::Object::is_null())
            {
                return std::move(lua::Object::template as<T>());
            }
            else
            {
                RUNTIME_ERROR("undefined lua object!");
            }
        }

        template<typename T>
        T as(T const &default_value) const
        {
            if (m_value_ != "")
            {
                return std::move(type_cast<T>(m_value_));
            }
            else if (!lua::Object::is_null())
            {
                return std::move(lua::Object::template as<T>(default_value));
            }
            else
            {
                return default_value;
            }
        }

        template<typename T>
        void as(T *v) const
        {
            *v = as<T>(*v);
        }

        template<typename T>
        operator T() const
        {
            return this->template as<T>();
        }

    private:
        std::string m_value_;
    };

    DictObject operator[](std::string const &key) const
    {

        auto it = m_kv_map_.find(key);
        if (it != m_kv_map_.end())
        {
            return std::move(DictObject(it->second));
        }
        else if (!m_lua_object_.is_null())
        {
            return std::move(DictObject(m_lua_object_[key]));
        }
        else
        {
            return std::move(DictObject());
        }
    }

private:

    lua::Object m_lua_object_;
    std::map<std::string, std::string> m_kv_map_;
};

}  // namespace simpla

#endif /* CORE_UTILITIES_CONFIG_PARSER_H_ */
