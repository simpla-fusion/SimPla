//
// Created by salmon on 16-10-7.
//

#ifndef SIMPLA_DATABASEANY_H
#define SIMPLA_DATABASEANY_H

#include <map>
#include <memory>
#include "../sp_config.h"
#include "DataBase.h"
#include "DataType.h"
#include "DataSpace.h"

namespace simpla { namespace toolbox
{


class DataBaseAny : public DataBase
{
public:
    struct Entity;
    struct iterator;
    struct const_iterator;

    DataBaseAny() {};

    ~DataBaseAny() {};

    void swap(DataBaseAny &other);

    std::ostream &print(std::ostream &os, int indent = 0) const;

    bool open(std::string path);

    void close();

    /**
     * as value
     * @{
     */

    bool is_table() const { return m_value_.is_null() && !m_table_.empty(); }

    bool has_value() const { return !m_value_.is_null(); };

    Entity const &value() const { return m_value_; }

    Entity &value() { return m_value_; }

    /** @} */

    /**
     *  as container
     *  @{
     */

    size_t size() const;

    bool empty() const;


    bool has(std::string const &key) const;

    iterator find(std::string const &key);

    std::pair<iterator, bool> insert(std::string const &, std::shared_ptr<DataBaseAny> &);

    /**
    *  if key exists then return ptr else create and return ptr
    * @param key
    * @return
    */
    std::shared_ptr<DataBase> get(std::string const &key);

    /**
     *  if key exists then return ptr else return null
     * @param key
     * @return
     */
    std::shared_ptr<DataBase> at(std::string const &key);

    std::shared_ptr<const DataBase> at(std::string const &key) const;

    iterator begin() { return m_table_.begin(); }

    iterator end() { return m_table_.end(); }

    const_iterator begin() const { return m_table_.cbegin(); }

    const_iterator end() const { return m_table_.end(); }

    /** @}*/

private:
    Entity m_value_;
    std::map<std::string, std::shared_ptr<DataBaseAny>> m_table_;

};

struct DataBaseAny::iterator
{

};

struct DataBaseAny::const_iterator
{

};

struct DataBaseAny::Entity
{
public:
    Entity() : m_holder_(nullptr) {}

    template<typename ValueType>
    Entity(const ValueType &value)
            : m_holder_(new Holder<typename std::remove_cv<typename std::decay<ValueType>::type>::type>(value)) {}

    template<typename ValueType>
    Entity(std::shared_ptr<ValueType>
           &value) {};

    Entity(const Entity &other) : m_holder_(other.m_holder_ == nullptr ? other.m_holder_->clone() : nullptr) {}

    // Move constructor
    Entity(Entity
           &&other) :
            m_holder_(other
                              .m_holder_) { other.m_holder_ = 0; }

    // Perfect forwarding of ValueType
    template<typename ValueType>
    Entity(ValueType
           &&value,
           typename std::enable_if<!(std::is_same<Entity &, ValueType>::value ||
                                     std::is_const<ValueType>::value)>::type * = 0
            // disable if entity has type `any&`
            // disable if entity has type `const ValueType&&`
    ) : m_holder_(new Holder<typename std::decay<ValueType>::type>(static_cast<ValueType &&>(value)))
    {
    }

    ~Entity() { delete m_holder_; }

    void swap(Entity &other)
    {
        std::swap(m_holder_, other.m_holder_);
    }

    Entity &operator=(const Entity &rhs)
    {
        Entity(rhs).swap(*this);
        return *this;
    }

    // move assignement
    Entity &operator=(Entity &&rhs)
    {
        rhs.swap(*this);
        Entity().swap(rhs);
        return *this;
    }

    // Perfect forwarding of ValueType
    template<class ValueType>
    Entity &operator=(ValueType &&rhs)
    {
        Entity(static_cast<ValueType &&>(rhs)).swap(*this);
        return *this;
    }

    bool empty() const { return m_holder_ == nullptr; }

    bool is_null() const { return m_holder_ == nullptr; }

    const void *data() const { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

    void *data() { return m_holder_ == nullptr ? nullptr : m_holder_->data(); };

    DataType data_type() { return m_holder_ == nullptr ? DataType() : m_holder_->data_type(); };

    DataSpace data_space() { return m_holder_ == nullptr ? DataSpace() : m_holder_->data_space(); };

    void clear() { Entity().swap(*this); }

    const std::type_info &type() const { return m_holder_ ? m_holder_->type() : typeid(void); }

    /** @} */

    template<class U> bool is_a() const { return m_holder_ != nullptr && m_holder_->type() == typeid(U); }

    template<class U> bool as(U *v) const
    {
        if (is_a<U>())
        {
            *v = dynamic_cast<Holder <U> *>(m_holder_)->value();
            return true;
        } else { return false; }
    }

    template<class U> operator U() const { return as<U>(); }

    template<class U> U const &as() const
    {
        if (!is_a<U>()) {THROW_EXCEPTION_BAD_CAST(typeid(U).name(), m_holder_->type().name()); }

        return dynamic_cast<Holder <U> const *>(m_holder_)->m_value_;
    }

    template<class U> U &as()
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

        ~PlaceHolder() {}

        PlaceHolder *clone() const = 0;

        const std::type_info &type() const = 0;

        std::ostream &print(std::ostream &os, int indent = 1) const = 0;

        void *data()=0;

        size_type size_in_byte() const = 0;

        DataType data_type() const =0;

        DataSpace const &data_space() const =0;
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
//
//struct DataFuction : public Entity
//{
//    /**
//      *  as function
//      *  @{
//      */
//protected:
//
//     Entity pop_return();
//
//     void push_parameter(Entity const &);
//
//private:
//    template<typename TFirst> inline void push_parameters(TFirst &&first) { push_parameter(Entity(first)); }
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
//    Entity call(Args &&...args)
//    {
//        push_parameters(std::forward<Args>(args)...);
//        return pop_return();
//    };
//
//    template<typename ...Args> inline
//    Entity operator()(Args &&...args) { return call(std::forward<Args>(args)...); };
//};
//struct DataBaseAny::iterator
//{
//    iterator() {};
//
//     ~iterator() {};
//
//     bool is_equal(iterator const &other) const;
//
//     std::pair<std::string, std::shared_ptr<DataBaseAny>> get() const;
//
//     iterator &next();
//
//     std::pair<std::string, std::shared_ptr<DataBaseAny>> value() const;
//
//    std::pair<std::string, std::shared_ptr<DataBaseAny>> operator*() const { return value(); };
//
//    std::pair<std::string, std::shared_ptr<DataBaseAny>> operator->() const { return value(); };
//
//    bool operator!=(iterator const &other) const { return !is_equal(other); };
//
//    iterator &operator++() { return next(); }
//};

std::ostream &operator<<(std::ostream &os, DataBaseAny const &prop) { return prop.print(os, 0); }

}}//namespace simpla{namespace toolbox{
#endif //SIMPLA_DATABASEANY_H
