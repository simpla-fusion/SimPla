//
// Created by salmon on 16-10-7.
//

#ifndef SIMPLA_DATABASEANY_H
#define SIMPLA_DATABASEANY_H

#include <simpla/SIMPLA_config.h>

#include <map>
#include <memory>

#include "DataBase.h"
#include "Log.h"

//#include "DataType.h"
//#include "DataSpace.h"

namespace simpla { namespace toolbox
{


struct DataEntityAny : public DataEntity
{
public:
    DataEntityAny() : m_holder_(nullptr) {}

    DataEntityAny(char const *s)
            : m_holder_(new Holder<std::string>(std::string(s))) {}


    template<typename ValueType>
    DataEntityAny(const ValueType &value)
            : m_holder_(new Holder<typename std::remove_cv<typename std::decay<ValueType>::type>::type>(value)) {}

    template<typename ValueType>
    DataEntityAny(std::shared_ptr<ValueType> &value) {};

    DataEntityAny(const DataEntityAny &other) : m_holder_(
            other.m_holder_ == nullptr ? other.m_holder_->clone() : nullptr) {}

    // Move constructor
    DataEntityAny(DataEntityAny &&other) : m_holder_(other.m_holder_) { other.m_holder_ = 0; }


    ~DataEntityAny() { delete m_holder_; }

    void swap(DataEntityAny &other)
    {
        std::swap(m_holder_, other.m_holder_);
    }

    DataEntityAny &operator=(const DataEntityAny &rhs)
    {
        DataEntityAny(rhs).swap(*this);
        return *this;
    }

    // move assignement
    DataEntityAny &operator=(DataEntityAny &&rhs)
    {
        rhs.swap(*this);
        DataEntityAny().swap(rhs);
        return *this;
    }

    // Perfect forwarding of ValueType
    template<class ValueType>
    DataEntityAny &operator=(ValueType &&rhs)
    {
        DataEntityAny(static_cast<ValueType &&>(rhs)).swap(*this);
        return *this;
    }

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(DataEntityAny) || DataEntity::is_a(t_id);
    }

    virtual bool is_table() const { return false; };

    const std::type_info &type() const { return m_holder_ ? m_holder_->type() : typeid(void); }

    bool empty() const { return m_holder_ == nullptr; }

    bool is_null() const { return m_holder_ == nullptr; }

    const void *data() const { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

    void *data() { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

//    DataType data_type() { return m_holder_ == nullptr ? DataType() : m_holder_->data_type(); };
//
//    DataSpace data_space() { return m_holder_ == nullptr ? DataSpace() : m_holder_->data_space(); };

    void clear() { DataEntityAny().swap(*this); }


    /** @} */

    template<class U>
    bool is_a() const { return m_holder_ != nullptr && m_holder_->type() == typeid(U); }

    template<class U>
    bool as(U *v) const
    {
        if (is_a<U>())
        {
            *v = dynamic_cast<Holder <U> *>(m_holder_)->value();
            return true;
        } else { return false; }
    }

    template<class U>
    operator U() const { return as<U>(); }

    template<class U>
    U const &as() const
    {
        if (!is_a<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), m_holder_->type().name()); }

        return dynamic_cast<Holder <U> const *>(m_holder_)->m_value_;
    }

    template<class U>
    U &as()
    {
        if (!is_a<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), m_holder_->type().name()); }
        return dynamic_cast<Holder <U> *>(m_holder_)->m_value_;
    }

    template<class U>
    U as(U const &def_v) const
    {
        if (is_a<U>()) { return dynamic_cast<Holder <U> const *>(m_holder_)->value(); }
        else { return def_v; }
    }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {
        if (m_holder_ != nullptr) { m_holder_->print(os, indent); }
        return os;
    }


private:
    struct PlaceHolder;

    template<typename>
    struct Holder;


    PlaceHolder *clone() const { return (m_holder_ != nullptr) ? m_holder_->clone() : nullptr; }

    PlaceHolder *m_holder_;


    struct PlaceHolder
    {
        PlaceHolder() {}

        virtual  ~PlaceHolder() {}

        virtual PlaceHolder *clone() const = 0;

        virtual const std::type_info &type() const = 0;

        virtual std::ostream &print(std::ostream &os, int indent = 1) const = 0;

        virtual void *data()=0;

//        virtual size_type size_in_byte() const = 0;
//        virtual DataType data_type() const =0;
//        virtual DataSpace const &data_space() const =0;
    };

    template<typename ValueType>
    struct Holder : PlaceHolder
    {
        ValueType m_value_;
        std::shared_ptr<DataSpace> m_space_;

        Holder(ValueType const &v) : m_value_(v) {}

        Holder(ValueType &&v) : m_value_(std::forward<ValueType>(v)) {}

        ~Holder() {}

        Holder &operator=(const Holder &) = delete;

        PlaceHolder *clone() const { return new Holder(m_value_); }

        const std::type_info &type() const { return typeid(ValueType); }


        std::ostream &print(std::ostream &os, int indent = 1) const
        {
            if (std::is_same<ValueType, std::string>::value) { os << "\"" << m_value_ << "\""; }
            else
            {
                os << m_value_;
            }
            return os;
        }

        void *data() { return &m_value_; };

        size_type size_in_byte() const { return sizeof(ValueType); }

//        DataType data_type() const { return DataType::template create<ValueType>(); }
//
//        DataSpace const &data_space() const { return *m_space_; }

        ValueType &value() { return m_value_; }

        ValueType const &value() const { return m_value_; }

    };

    template<typename ValueType>
    struct Holder<std::shared_ptr<ValueType>> : PlaceHolder
    {
        std::shared_ptr<ValueType> m_value_;
        std::shared_ptr<DataSpace> m_space_;

        Holder(std::shared_ptr<ValueType> v, std::shared_ptr<DataSpace> sp) : m_value_(v), m_space_(sp) {}

        ~Holder() {}

        Holder &operator=(const Holder &) = delete;

        PlaceHolder *clone() const
        {
            return new Holder(std::shared_ptr<ValueType>(malloc(size_in_byte()), m_space_));
        }

        const std::type_info &type() const { return typeid(ValueType); }

        void *data() { return m_value_.get(); };

        size_type size_in_byte() const { return 0;/*sizeof(ValueType) * m_space_.size();*/ }
//
//        DataType data_type() const { return DataType::template create<ValueType>(); }
//
//        DataSpace const &data_space() const { return m_space_; }

        ValueType *value() { return m_value_.get(); }

        ValueType const *value() const { return m_value_.get(); }

        std::ostream &print(std::ostream &os, int indent = 1) const
        {
//            UNIMPLEMENTED;
            return os;
        }
    };
};

class DataBaseMemory : public DataBase
{
public:

    DataBaseMemory() {};

    ~DataBaseMemory() {};

    bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(DataBaseMemory) || DataBase::is_a(t_id);
    };

    void swap(DataBaseMemory &other);

//    bool eval(std::string path) { return true; };
//
//    bool open(std::string const &path, int);
//
//    void close();

    /**
     * as value
     * @{
     */



    template<typename T>
    T const &as(std::string const &key) const
    {
        return std::dynamic_pointer_cast<DataEntityAny>(m_table_.at(key))->as<T>();
    }

    std::shared_ptr<DataBaseMemory>
    create(std::string const &key)
    {
        auto res = std::make_shared<DataBaseMemory>();
        m_table_[key] = std::dynamic_pointer_cast<DataEntity>(res);

        return res;
    }

    /** @} */

    /**
     *  as container
     *  @{
     */

    size_t size() const;

    bool empty() const;

    bool has(std::string const &key) const;


    void set(std::string const &, std::shared_ptr<DataEntity> const &);

    void set(std::string const &, std::shared_ptr<DataEntityAny> const &);

    void set(std::string const &, std::shared_ptr<DataBaseMemory> const &);

    template<typename T>
    bool set(std::string const &key, T const &v) { set(key, std::make_shared<DataEntityAny>(v)); };


    std::shared_ptr<DataEntity> get(std::string const &key);

    std::shared_ptr<DataEntity> at(std::string const &key);

    std::shared_ptr<const DataEntity> at(std::string const &key) const;

    void foreach(std::function<void(std::string const &, DataEntity &)> const &fun);

    void foreach(std::function<void(std::string const &, DataEntity const &)> const &fun) const;

    /** @}*/


private:
    std::map<std::string, std::shared_ptr<DataEntity>> m_table_;

};

//
//struct DataFuction : public DataEntityAny
//{
//    /**
//      *  as function
//      *  @{
//      */
//protected:
//
//     DataEntityAny pop_return();
//
//     void push_parameter(DataEntityAny const &);
//
//private:
//    template<typename TFirst> inline void push_parameters(TFirst &&first) { push_parameter(DataEntityAny(first)); }
//
//    template<typename TFirst, typename ...Args> inline
//    void push_parameters(TFirst &&first, Args &&...args)
//    {
//        push_parameters(std::forward<TFirst>(first));
//        push_parameters(std::forward<Args>(args)...);
//    };
//public:
//
//    template<typename ...Args> inline
//    DataEntityAny call(Args &&...args)
//    {
//        push_parameters(std::forward<Args>(args)...);
//        return pop_return();
//    };
//
//    template<typename ...Args> inline
//    DataEntityAny operator()(Args &&...args) { return call(std::forward<Args>(args)...); };
//};
//struct DataBaseMemory::iterator
//{
//    iterator() {};
//
//     ~iterator() {};
//
//     bool is_equal(iterator const &other) const;
//
//     std::pair<std::string, std::shared_ptr<DataBaseMemory>> get() const;
//
//     iterator &next();
//
//     std::pair<std::string, std::shared_ptr<DataBaseMemory>> value() const;
//
//    std::pair<std::string, std::shared_ptr<DataBaseMemory>> operator*() const { return value(); };
//
//    std::pair<std::string, std::shared_ptr<DataBaseMemory>> operator->() const { return value(); };
//
//    bool operator!=(iterator const &other) const { return !is_equal(other); };
//
//    iterator &operator++() { return next(); }
//};

//std::ostream &operator<<(std::ostream &os, DataBaseMemory const &prop) { return prop.print(os, 0); }

}}//namespace simpla{namespace toolbox{
#endif //SIMPLA_DATABASEANY_H
