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

#include "../data_model/DataType.h"
#include "utilities/log.h"

namespace simpla
{

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

    any(any &that) : ptr_(that.clone()) { }

    any(any const &that) : ptr_(that.clone()) { }

    any(any &&that) : ptr_(std::move(that.ptr_)) { }

    void swap(any &other) { std::swap(ptr_, other.ptr_); }

    virtual bool empty() const { return !bool(ptr_); }

    virtual bool IsNull() const { return empty(); }

    operator bool() const { return !empty(); }

    void const *data() const { return ptr_->data(); }

    void *data() { return ptr_->data(); }

    std::string string() const
    {
        std::ostringstream os;
        this->print(os, 0);
        return os.str();
    }

    template<class U> bool is_same() const { return ptr_ != nullptr && ptr_->is_same<U>(); }

    bool is_boolean() const { return ptr_ != nullptr && ptr_->is_boolean(); }

    bool is_integral() const { return ptr_ != nullptr && ptr_->is_integral(); }

    bool is_floating_point() const { return ptr_ != nullptr && ptr_->is_floating_point(); }

    bool is_string() const { return ptr_ != nullptr && ptr_->is_string(); }

    template<class U>
    bool as(U *v) const
    {
        bool is_found = false;
        if (is_same<U>())
        {
            *v = dynamic_cast<Derived <U> *>(ptr_.get())->m_value;
            is_found = true;
        }
        return is_found;
    }

    template<class U>
    U const &as() const
    {
        if (!is_same<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), ptr_->type_name()); }

        return dynamic_cast<Derived <U> *>(ptr_.get())->m_value;
    }

    template<class U>
    U &as()
    {

        if (!is_same<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), ptr_->type_name()); }

        return dynamic_cast<Derived <U> *>(ptr_.get())->m_value;
    }

    template<class U>
    U as(U const &def_v) const
    {

        if (empty() || !is_same<U>())
        {
            return def_v;
        }
        else
        {
            return dynamic_cast<Derived <U> *>(ptr_.get())->m_value;
        }
    }

    template<class U> operator U() const { return as<U>(); }

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

    data_model::DataType datatype() const { return ptr_->data_type(); }

private:
    struct Base;
    typedef std::unique_ptr<Base> BasePtr;

    struct Base
    {
        virtual ~Base() { }

        virtual BasePtr clone() const = 0;

        virtual void const *data() const = 0;

        virtual void *data() = 0;

        virtual std::ostream &print(std::ostream &os, int indent = 0) const = 0;

        virtual bool is_same(std::type_index const &) const = 0;

        virtual std::string type_name() const = 0;

        template<typename T> bool is_same() const { return is_same(std::type_index(typeid(T))); }

        virtual data_model::DataType data_type() const = 0;


        virtual bool is_boolean() const = 0;

        virtual bool is_integral() const = 0;

        virtual bool is_floating_point() const = 0;

        virtual bool is_string() const = 0;

    };

    template<typename T>
    struct Derived : Base
    {
        template<typename U>
        Derived(U &&value) : m_value(std::forward<U>(value)) { }

        BasePtr clone() const { return BasePtr(new Derived<T>(m_value)); }

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

        virtual bool is_boolean() const { return std::is_same<T, bool>::value; }

        virtual bool is_integral() const { return std::is_integral<T>::value; }

        virtual bool is_floating_point() const { return std::is_floating_point<T>::value; }

        virtual bool is_string() const { return std::is_same<T, std::string>::value; }

        T m_value;

    };

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
