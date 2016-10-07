//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DICT_H
#define SIMPLA_DICT_H

#include <map>
#include "../sp_config.h"
#include "DataEntity.h"

namespace simpla { namespace toolbox
{
class DataEntity
{
public:
    DataEntity() : m_holder_(nullptr) {}

    template<typename ValueType>
    DataEntity(const ValueType &value)
            : m_holder_(new Holder<typename std::remove_cv<typename std::decay<ValueType>::type>::type>(value)) {}

    template<typename ValueType>
    DataEntity(std::shared_ptr<ValueType> &value) {};

    DataEntity(const DataEntity &other) : m_holder_(other.m_holder_ == nullptr ? other.m_holder_->clone() : nullptr) {}

    // Move constructor
    DataEntity(DataEntity &&other) : m_holder_(other.m_holder_) { other.m_holder_ = 0; }

    // Perfect forwarding of ValueType
    template<typename ValueType>
    DataEntity(ValueType &&value,
               typename std::enable_if<!(std::is_same<DataEntity &, ValueType>::value ||
                                         std::is_const<ValueType>::value)>::type * = 0
            // disable if entity has type `any&`
            // disable if entity has type `const ValueType&&`
    ) : m_holder_(new Holder<typename std::decay<ValueType>::type>(static_cast<ValueType &&>(value)))
    {
    }

    ~DataEntity() { delete m_holder_; }

    DataEntity &swap(DataEntity &other)
    {
        std::swap(m_holder_, other.m_holder_);
        return *this;
    }

    DataEntity &operator=(const DataEntity &rhs) { return DataEntity(rhs).swap(*this); }

    // move assignement
    DataEntity &operator=(DataEntity &&rhs)
    {
        rhs.swap(*this);
        DataEntity().swap(rhs);
        return *this;
    }

    // Perfect forwarding of ValueType
    template<class ValueType>
    DataEntity &operator=(ValueType &&rhs) { return DataEntity(static_cast<ValueType &&>(rhs)).swap(*this); }

    virtual bool empty() const { return m_holder_ == nullptr; }

    virtual bool is_null() const { return m_holder_ == nullptr; }

    virtual const void *data() const { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

    virtual void *data() { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

    virtual DataType data_type() { return m_holder_ == nullptr ? DataType() : m_holder_->data_type(); };

    virtual DataSpace const &data_space() { return m_holder_ == nullptr ? DataSpace() : m_holder_->data_space(); };

    void clear() { DataEntity().swap(*this); }

    const std::type_info &type() const { return m_holder_ ? m_holder_->type() : typeid(void); }

    operator bool() const { return !is_null(); }

    operator std::string() const
    {
        std::ostringstream os;
        this->print(os, 0);
        return os.str();
    }

    /** @} */

    template<class U> bool is_a() const { return m_holder_ != nullptr && m_holder_->type() == typeid(U); }

    template<class U> bool as(U *v) const
    {
        if (is_a<U>())
        {
            *v = dynamic_cast<Holder <U> *>(m_holder_)->value();
            return true;
        } else
        {
            return false;
        }
    }

    template<class U> operator U() const { return as<U>(); }

    template<class U> U const &as() const
    {
        if (!is_a<U>()) { THROW_EXCEPTION_BAD_CAST(typeid(U).name(), m_holder_->type().name()); }

        return dynamic_cast<Holder <U> const *>(m_holder_)->m_value_;
    }

    template<class U> U &as()
    {
        if (!is_a<U>()) { THROW_EXCEPTION_BAD_CAST(typeid(U).name(), m_holder_->type().name()); }
        return dynamic_cast<Holder <U> *>(m_holder_)->m_value_;
    }

    template<class U>
    U as(U const &def_v) const
    {
        if (is_a<U>()) { return dynamic_cast<Holder <U> const *>(m_holder_)->value(); }
        else { return def_v; }
    }


    std::ostream &print(std::ostream &os, int indent = 1) const
    {
        if (m_holder_ != nullptr) { m_holder_->print(os, indent); }
        return os;
    }


private:
    struct PlaceHolder;
    template<typename> struct Holder;


    PlaceHolder *clone() const { return (m_holder_ != nullptr) ? m_holder_->clone() : nullptr; }

    PlaceHolder *m_holder_;


    struct PlaceHolder
    {
        PlaceHolder() {}

        virtual ~PlaceHolder() {}

        virtual PlaceHolder *clone() const = 0;

        virtual const std::type_info &type() const = 0;

        virtual std::ostream &print(std::ostream &os, int indent = 1) const = 0;

        virtual void *data()=0;

        virtual size_type size_in_byte() const = 0;

        virtual DataType data_type() const =0;

        virtual DataSpace const &data_space() const =0;
    };

    template<typename ValueType>
    struct Holder : PlaceHolder
    {
        ValueType m_value_;
        static DataSpace m_space_;

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

        void *data() { return &m_holder_; };

        size_type size_in_byte() const { return sizeof(ValueType); }

        DataType data_type() const { return DataType::template create<ValueType>(); }

        DataSpace const &data_space() const { return m_space_; }

        ValueType &value() { return m_value_; }

        ValueType const &value() const { return m_value_; }

    };

    template<typename ValueType>
    struct Holder<std::shared_ptr<ValueType>> : PlaceHolder
    {
        std::shared_ptr<ValueType> m_value_;
        DataSpace m_space_;

        Holder(std::shared_ptr<ValueType> v, DataSpace const &sp) : m_value_(v), m_space_(sp) {}

        ~Holder() {}

        Holder &operator=(const Holder &) = delete;

        PlaceHolder *clone() const
        {
            return new Holder(std::shared_ptr<ValueType>(malloc(size_in_byte()), m_space_));
        }

        const std::type_info &type() const { return typeid(ValueType); }

        void *data() { return m_value_.get(); };

        size_type size_in_byte() const { return sizeof(ValueType) * m_space_.size(); }

        DataType data_type() const { return DataType::template create<ValueType>(); }

        DataSpace const &data_space() const { return m_space_; }

        ValueType *value() { return m_value_; }

        ValueType const *value() const { return m_value_; }

        std::ostream &print(std::ostream &os, int indent = 1) const
        {
            UNIMPLEMENTED;
            return os;
        }
    };
};

class DataBase
{
public:

    DataBase() {};

    virtual  ~DataBase() {};

    virtual void swap(DataBase &other);

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    /**
     * as value
     * @{
     */

    virtual bool is_table() const;

    virtual bool has_value() const { return !m_value_.is_null(); };

    virtual DataEntity const &value() const { return m_value_; }

    virtual DataEntity &value() { return m_value_; }

    /** @} */

    /**
     *  as container
     *  @{
     */

    virtual size_t size() const;

    virtual bool empty() const;

    virtual bool has_a(std::string const &key) const;

    typedef typename std::map<std::string, std::shared_ptr<DataBase>>::iterator iterator;

    virtual std::pair<iterator, bool> insert(std::string const &, std::shared_ptr<DataBase> &);

    virtual std::shared_ptr<DataBase> at(std::string const &);

    virtual std::shared_ptr<const DataBase> at(std::string const &) const;

    std::shared_ptr<DataBase> operator[](std::string const &key) { return at(key); };

    std::shared_ptr<const DataBase> operator[](std::string const &key) const { return at(key); };
//    struct iterator;

    virtual iterator begin();

    virtual iterator end();

    virtual iterator begin() const;

    virtual iterator end() const;
    /** @}*/

private:
    DataEntity m_value_;
    std::map<std::string, std::shared_ptr<DataBase>> m_table_;

};

//
//struct DataFuction : public DataEntity
//{
//    /**
//      *  as function
//      *  @{
//      */
//protected:
//
//    virtual DataEntity pop_return();
//
//    virtual void push_parameter(DataEntity const &);
//
//private:
//    template<typename TFirst> inline void push_parameters(TFirst &&first) { push_parameter(DataEntity(first)); }
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
//    DataEntity call(Args &&...args)
//    {
//        push_parameters(std::forward<Args>(args)...);
//        return pop_return();
//    };
//
//    template<typename ...Args> inline
//    DataEntity operator()(Args &&...args) { return call(std::forward<Args>(args)...); };
//};
//struct DataBase::iterator
//{
//    iterator() {};
//
//    virtual ~iterator() {};
//
//    virtual bool is_equal(iterator const &other) const;
//
//    virtual std::pair<std::string, std::shared_ptr<DataBase>> get() const;
//
//    virtual iterator &next();
//
//    virtual std::pair<std::string, std::shared_ptr<DataBase>> value() const;
//
//    std::pair<std::string, std::shared_ptr<DataBase>> operator*() const { return value(); };
//
//    std::pair<std::string, std::shared_ptr<DataBase>> operator->() const { return value(); };
//
//    bool operator!=(iterator const &other) const { return !is_equal(other); };
//
//    iterator &operator++() { return next(); }
//};

std::ostream &operator<<(std::ostream &os, DataBase const &prop) { return prop.print(os, 0); }

}}//namespace simpla{namespace toolbox{
#endif //SIMPLA_DICT_H
