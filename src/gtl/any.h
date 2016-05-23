/**
 * @file any.h
 *
 * @date    2014-7-13  AM7:18:19
 * @author salmon
 */

#ifndef ANY_H_
#define ANY_H_

#include <algorithm>
#include <cstdbool>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <stddef.h>

#include "../data_model/DataType.h"
#include "Log.h"

namespace simpla
{
struct Base;

template<typename> class Derived;

struct any;

namespace _impl
{

template<typename T>
auto _to_bool(T const &v)
-> typename std::enable_if<std::is_convertible<T, bool>::value, bool>::type { return static_cast<bool>(v); };

template<typename T>
auto _to_bool(T const &v)
-> typename std::enable_if<!std::is_convertible<T, bool>::value, bool>::type { return false; };


template<typename T>
auto _to_integer(T const &v)
-> typename std::enable_if<std::is_convertible<T, int>::value, int>::type
{
    return static_cast<int>(v);
};

template<typename T>
auto _to_integer(T const &v)
-> typename std::enable_if<!std::is_convertible<T, int>::value, int>::type { return 0; };


template<typename T>
auto _to_floating_point(T const &v)
-> typename std::enable_if<std::is_convertible<T, double>::value, double>::type { return static_cast<double>(v); };

template<typename T>
auto _to_floating_point(T const &v)
-> typename std::enable_if<!std::is_convertible<T, double>::value, double>::type { return 0; };

inline std::string _to_string(std::string const &v) { return (v); };

template<typename T>
std::string _to_string(T const &v) { return ""; };

template<typename T>
auto get_bool(T *v, bool other)
-> typename std::enable_if<std::is_integral<T>::value, bool>::type
{
    *v = other;
    return true;
}

template<typename T>
auto get_bool(T *v, bool)
-> typename std::enable_if<!std::is_integral<T>::value, bool>::type
{
    return false;
}


template<typename T>
auto get_integer(T *v, int other)
-> typename std::enable_if<std::is_integral<T>::value, bool>::type
{
    *v = other;
    return true;
}

template<typename T>
auto get_integer(T *v, int)
-> typename std::enable_if<!std::is_integral<T>::value, bool>::type
{
    return false;
}

template<typename T>
auto get_floating_point(T *v, double other)
-> typename std::enable_if<std::is_floating_point<T>::value, bool>::type
{
    *v = other;
    return true;
}

template<typename T>
auto get_floating_point(T *v, double)
-> typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
{
    return false;
}


template<typename T>
bool get_string(T *v, std::string const &)
{
    return false;
}

inline bool get_string(std::string *v, std::string const &other)
{
    *v = other;
    return true;
}


}
struct Base
{
    virtual ~Base() { }

    virtual std::unique_ptr<Base> clone() const = 0;

    virtual void const *data() const = 0;

    virtual void *data() = 0;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const = 0;

    virtual bool is_same(std::type_index const &) const = 0;

    virtual std::string type_name() const = 0;

    template<typename T> bool is_same() const { return is_same(std::type_index(typeid(T))); }

    virtual data_model::DataType data_type() const = 0;


    virtual bool is_bool() const = 0;

    virtual bool is_integral() const = 0;

    virtual bool is_floating_point() const = 0;

    virtual bool is_string() const = 0;

    virtual bool to_bool() const = 0;

    virtual int to_integer() const = 0;

    virtual double to_floating_point() const = 0;

    virtual std::string to_string() const = 0;

    virtual int size() const = 0;

    virtual std::shared_ptr<Base> get(int) const = 0;

    template<typename U, int N>
    bool as(nTuple<U, N> *v) const
    {
        if (is_same<nTuple<U, N>>())
        {
            *v = dynamic_cast<Derived<nTuple<U, N>> const *>(this)->m_value;
            return true;
        }
        else if (this->size() < N)
        {
            return false;
        }

        for (int i = 0; i < N; ++i)
        {
            this->get(i)->as(&((*v)[i]));
        }

        return true;
    }


    template<class U>
    bool as(U *v) const
    {
        bool success = true;
        if (is_same<U>()) { *v = dynamic_cast<Derived<U> const *>(this)->m_value; }
        else if (_impl::get_integer(v, this->to_integer())) { }
        else if (_impl::get_floating_point(v, this->to_floating_point())) { }
        else if (_impl::get_string(v, this->to_string())) { }
        else { success = false; }

        return success;
    }
};

template<typename T>
struct Derived : Base
{
    template<typename U>
    Derived(U const &value) : m_value(value) { }

    template<typename U>
    Derived(U &&value) : m_value(std::forward<U>(value)) { }

    std::unique_ptr<Base> clone() const { return std::unique_ptr<Base>(new Derived<T>(m_value)); }

    void const *data() const { return reinterpret_cast<void const *>(&m_value); }

    void *data() { return reinterpret_cast<void *>(&m_value); }

    std::ostream &print(std::ostream &os, int indent = 0) const
    {
        if (std::is_same<T, std::string>::value) { os << "\"" << m_value << "\""; } else { os << m_value; }

        return os;
    }

    bool is_same(std::type_index const &t_idx) const { return std::type_index(typeid(T)) == t_idx; }

    std::string type_name() const { return typeid(T).name(); }

    data_model::DataType data_type() const { return data_model::DataType::template create<T>(); }

    virtual bool is_bool() const { return std::is_convertible<T, bool>::value; }

    virtual bool is_integral() const { return std::is_convertible<T, int>::value; }

    virtual bool is_floating_point() const { return std::is_convertible<T, double>::value; }

    virtual bool is_string() const { return std::is_convertible<T, std::string>::value; }


public:

    virtual int to_integer() const { return _impl::_to_integer(m_value); };

    virtual double to_floating_point() const { return _impl::_to_floating_point(m_value); };

    virtual bool to_bool() const { return _impl::_to_bool(m_value); };

    virtual std::string to_string() const { return _impl::_to_string(m_value); };


private:
    template<typename V> int _size_of(V const &) const { return 1; }

    template<typename V, int N, int ...M> int _size_of(nTuple<V, N, M...> const &) const { return N; }

    template<typename ...V> int _size_of(std::tuple<V ...> const &) const { return sizeof...(V); }


    template<typename V> std::shared_ptr<Base> _index_of(V const &v, int n) const
    {
        return std::shared_ptr<Base>(new Derived<V>(v));
    }

    template<typename V, int N>
    std::shared_ptr<Base> _index_of(nTuple<V, N> const &v, int n) const
    {
        return std::shared_ptr<Base>(new Derived<V>(v[n]));
    }

    template<typename T0, typename T1> std::shared_ptr<Base> _index_of(std::tuple<T0, T1> const &v, int n) const
    {
        std::shared_ptr<Base> res;
        switch (n)
        {
            case 0:
                res = std::shared_ptr<Base>(new Derived<T0>(std::get<0>(v)));
            case 1:
                res = std::shared_ptr<Base>(new Derived<T1>(std::get<1>(v)));
            default :
                OUT_OF_RANGE << n << " >  2 " << std::endl;
        }

        return res;
    }


//    template<typename T0, typename T1, typename T2> any _index_of(std::tuple<T0, T1, T2> const &v, int n) const
//    {
//        switch (n)
//        {
//            case 0:
//                return any(std::get<0>(v));
//            case 1:
//                return any(std::get<1>(v));
//            case 2:
//                return any(std::get<2>(v));
//            default :
//                OUT_OF_RANGE << n << " >  3 " << std::endl;
//        }
//    }
//
//    template<typename T0, typename T1, typename T2, typename T3>
//    any _index_of(std::tuple<T0, T1, T2, T3> const &v, int n) const
//    {
//        switch (n)
//        {
//            case 0:
//                return any(std::get<0>(v));
//            case 1:
//                return any(std::get<1>(v));
//            case 2:
//                return any(std::get<2>(v));
//            case 3:
//                return any(std::get<3>(v));
//            default :
//                OUT_OF_RANGE << n << " >  4 " << std::endl;
//        }
//    }


public:
    virtual int size() const { return _size_of(m_value); };

    virtual std::shared_ptr<Base> get(int n) const { return _index_of(m_value, n); }

    T m_value;

};


/**
 *  @ingroup gtl
 *   Base on http://www.cnblogs.com/qicosmos/p/3420095.html
 *   alt. <boost/any.hpp>
 *
 *   This an implement of 'any' with data type description/serialization information
 */
struct any
{
    template<typename U>
    any(U &&value) : ptr_(new Derived<typename std::decay<U>::type>(std::forward<U>(value))) { }

    any(void) { }

    any(any &other) : ptr_(other.clone()) { }

    any(any const &other) : ptr_(other.clone()) { }

    void swap(any &other) { std::swap(ptr_, other.ptr_); }

    virtual bool empty() const { return ptr_ == nullptr; }


    operator bool() const { return ptr_ != nullptr; }

    void const *data() const { return ptr_ != nullptr ? ptr_->data() : nullptr; }

    void *data() { return ptr_ != nullptr ? ptr_->data() : nullptr; }

    std::string string() const
    {
        std::ostringstream os;
        this->print(os, 0);
        return os.str();
    }

    template<class U> bool is_same() const { return ptr_ != nullptr && ptr_->is_same<U>(); }

    bool is_boolean() const { return ptr_ != nullptr && ptr_->is_bool(); }

    bool is_integral() const { return ptr_ != nullptr && ptr_->is_integral(); }

    bool is_floating_point() const { return ptr_ != nullptr && ptr_->is_floating_point(); }

    bool is_string() const { return ptr_ != nullptr && ptr_->is_string(); }

    template<class U> bool as(U *v) const { return ptr_ != nullptr && ptr_->as(v); }

    template<class U> operator U() const { return as<U>(); }

    template<class U> U as() const
    {
        U res;

        as(&res);

        return std::move(res);
    }

    template<class U>
    U as(U const &def_v) const
    {
        if (!empty() && this->template is_same<U>())
        {
            return dynamic_cast<Derived<U> *>(ptr_.get())->m_value;
        }
        else
        {
            U res;
            if (as(&res)) { return std::move(res); }
            else { return def_v; }
        }

    }


    template<class U> U const &get() const
    {
        if (!is_same<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), ptr_->type_name()); }

        return dynamic_cast<Derived<U> *>(ptr_.get())->m_value;
    }

    template<class U> U &get()
    {
        if (!is_same<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), ptr_->type_name()); }
        return dynamic_cast<Derived<U> *>(ptr_.get())->m_value;
    }


    any &operator=(const any &a)
    {
        if (ptr_ == a.ptr_) return *this;

        ptr_ = a.clone();

        return *this;
    }

    template<typename T>
    any &operator=(T const &v)
    {
        if (is_same<T>()) { as<T>() = v; }
        else { any(v).swap(*this); }
        return *this;
    }

    std::ostream &print(std::ostream &os, int indent = 0) const
    {
        if (ptr_ != nullptr) { ptr_->print(os, indent); }
        return os;
    }

    data_model::DataType data_type() const { return ptr_->data_type(); }

private:

    typedef std::unique_ptr<Base> BasePtr;


    BasePtr clone() const
    {
        if (ptr_ != nullptr) return ptr_->clone();

        return nullptr;
    }

    BasePtr ptr_;


};

}
// namespace simpla

#endif /* ANY_H_ */
