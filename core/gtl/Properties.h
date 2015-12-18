/**
 * @file   properties.h
 *
 * @date    2014-7-13  AM7:27:37
 * @author salmon
 */

#ifndef PROPERTIES_H_
#define PROPERTIES_H_

#include <map>
#include <string>
#include "ntuple.h"
#include "any.h"

namespace simpla
{

/**
 *  @ingroup gtl
 *  @{
 *  @brief Properties Tree
 *  @todo using shared_ptr storage data
 */
class Properties : public any, public std::map<std::string, Properties>
{

private:

    typedef Properties this_type;

    typedef std::string key_type;
    typedef std::map<key_type, this_type> map_type;

    bool is_changed_ = false;
public:
    Properties()
    {
    }

    template<typename T>
    Properties(T const &v) :
            value_type(v)
    {
    }

    ~Properties()
    {
    }

    this_type &operator=(this_type const &other)
    {
        any(dynamic_cast<any const &>(other)).swap(*this);
        map_type(dynamic_cast<map_type const &>(other)).swap(*this);

//		map_type(other).swap(*this);
        return *this;
    }

    template<typename T>
    this_type &operator=(T const &v)
    {

        any(v).swap(*this);
        return *this;
    }

    inline bool empty() const // STL style
    {
        return any::empty() && map_type::empty();
    }

    inline bool IsNull() const
    {
        return empty();
    }

    inline bool is_changed() const
    {
        return is_changed_;
    }

    void update()
    {
        is_changed_ = false;
    }

    operator bool() const
    {
        return !empty();
    }

    Properties &get(std::string const &key)
    {
        is_changed_ = true;
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

//	template<typename T>
//	auto as()
//			DECL_RET_TYPE((value_type::template as<typename array_to_ntuple_convert<T>::type>()))
//
//	template<typename T>
//	auto as() const
//			DECL_RET_TYPE((value_type::template as<typename array_to_ntuple_convert<T>::type>()))
//
//	template<typename T>
//	typename array_to_ntuple_convert<T>::type as(T const & default_v) const
//	{
//		typename array_to_ntuple_convert<T>::type res = default_v;
//		if (!value_type::empty())
//		{
//			res = value_type::template as<
//					typename array_to_ntuple_convert<T>::type>();
//		}
//
//		return std::move(res);
//	}

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

    template<typename T>
    void set(std::string const &key, T const &v)
    {
        get(key) = v;
    }

    template<typename T>
    void operator()(std::string const &key, T &&v)
    {
        set(key, std::forward<T>(v));
    }

    void operator()(Properties const &other)
    {
        append(other);
    }

    inline Properties &operator[](key_type const &key)
    {
        return get(key);
    }

    inline Properties &operator[](const char key[])
    {
        return get(key);
    }

    Properties const &operator()(std::string const &key) const
    {
        return get(key);
    }

    Properties &operator()(std::string const &key)
    {
        return get(key);
    }

    Properties const &operator()() const
    {
        return *this;
    }

    Properties &operator()()
    {
        return *this;
    }

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

    virtual inline std::ostream &print(std::ostream &os, int indent = 0) const
    {
        dynamic_cast<any const &>(*this).print(os, indent);
        for (auto const &item : *this)
        {
            os << item.first << " =  ";
            item.second.print(os, indent + 1);
            os << " , ";
            if (item.second.size() > 0)
            {
                item.second.print(os, indent + 1);
            }
        }
        return os;
    }
};

#define HAS_PROPERTIES                                       \
 inline Properties const& properties()const{return m_properties_;}  \
 inline Properties  & properties() {touch();return m_properties_;}  \
 private: Properties m_properties_; public:

#define DEFINE_PROPERTIES(_TYPE_, _NAME_)                                                      \
void _NAME_(_TYPE_ const & v)   \
{m_##_NAME_##_ =v; this->m_properties_[__STRING(_NAME_)] = v; this->touch();}       \
_TYPE_ _NAME_()const{return m_##_NAME_##_;}                         \
private: _TYPE_ m_##_NAME_##_;    public:
/** @} */
}  // namespace simpla

#endif /* PROPERTIES_H_ */
