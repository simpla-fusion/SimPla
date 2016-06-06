/**
 * @file   properties.h
 *
 * @date    2014-7-13  AM7:27:37
 * @author salmon
 */

#ifndef SIMPLA_GTL_PROPERTIES_H_
#define SIMPLA_GTL_PROPERTIES_H_

#include <map>
#include <string>
#include <iomanip>
#include "nTuple.h"
#include "any.h"
#include "../base/Object.h"

namespace simpla
{

/**
 *  @ingroup gtl
 *  @{
 *  @brief Properties Tree
 *  @todo using shared_ptr storage m_data
 */
class Properties
        : public any,
          public std::map<std::string, Properties>,
          public base::Object
{

private:

    typedef Properties this_type;

    typedef std::string key_type;
    typedef std::map<key_type, this_type> map_type;

public:

    Properties() { }

    Properties(this_type const &other) : any(dynamic_cast<any const &>(other)), map_type(other) { }

    template<typename T> Properties(T const &v) : any(v) { }

    ~Properties() { }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    void swap(this_type &other)
    {
        any::swap(other);
        map_type::swap(other);
    }

    template<typename T>
    this_type &operator=(T const &v)
    {
        any(v).swap(*this);
        return *this;
    }


    inline bool empty() const { return any::empty() && map_type::empty(); }

    inline bool has(std::string const &s) const { return map_type::find(s) != map_type::end(); }

    operator bool() const { return !empty() && (this->as<bool>()); }

    Properties &get(std::string const &key)
    {
        if (key == "")
        {
            return *this;
        }
        else
        {
            return map_type::operator[](key);
        }
    }

    Properties const &get(std::string const &key) const
    {
        auto it = map_type::find(key);
        if (it == map_type::end())
        {
            return *this;
        }
        else
        {
            return it->second;
        }
    }


    template<typename T>
    T get(std::string const &key, T const &default_v) const
    {
        T res = default_v;

        auto it = map_type::find(key);

        if (it != map_type::end() && it->second.is_same<T>())
        {
            res = it->second.template as<T>();
        }
        return std::move(res);
    }

    template<typename T>
    bool get(std::string const &key, T *v) const
    {
        bool is_found = false;

        auto it = map_type::find(key);

        if (it != map_type::end() && it->second.is_same<T>())
        {
            *v = it->second.template as<T>();

            is_found = true;
        }
        return is_found;
    }

    template<typename T> void set(std::string const &key, T const &v) { get(key) = v; }


    inline Properties &operator[](key_type const &key) { return get(key); }

    inline Properties &operator[](const char key[]) { return get(key); }

//   template<typename T>   void operator()(std::string const &key, T &&v) { set(key, std::forward<T>(v)); }
//
//    void operator()(Properties const &other) { append(other); }
//
//    Properties const &operator()(std::string const &key) const { return get(key); }
//
//    Properties &operator()(std::string const &key) { return get(key); }
//
//    Properties const &operator()() const   {  return *this; }
//
//    Properties &operator()() {   return *this;  }

    inline Properties const &operator[](key_type const &key) const
    {
        return get(key);
    }

    inline Properties const &operator[](const char key[]) const
    {
        return get(key);
    }

    this_type &append(this_type const &other)
    {
        for (auto const &item : other)
        {
            map_type::operator[](item.first) = (item.second);
        }
        return *this;
    }

    std::ostream &print(std::ostream &os, int indent = 1) const;
};

std::ostream &operator<<(std::ostream &os, Properties const &prop);


#define HAS_PROPERTIES                                                                                            \
virtual Properties const&properties()const {return m_properties_;};                                                         \
virtual Properties &properties()  {return m_properties_;};                                             \
private: Properties m_properties_; public:


#define DEFINE_PROPERTIES(_TYPE_, _NAME_)                                                      \
void _NAME_(_TYPE_ const & v)   \
{m_##_NAME_##_ =v; this->properties()[__STRING(_NAME_)] = v; }       \
_TYPE_ _NAME_()const{return m_##_NAME_##_;}                         \
private: _TYPE_ m_##_NAME_##_;    public:
/** @} */
}   // namespace simpla

#endif /* SIMPLA_GTL_PROPERTIES_H_ */
