//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/toolbox/Any.h>
#include <simpla/concept/Object.h>


namespace simpla { namespace data
{
struct DataEntityTable;
struct DataEntityLight;
struct DataEntityHeavy;

struct DataEntity : public concept::Printable
{
public:
    DataEntity() {}

    DataEntity(DataEntity const &other) {}

    DataEntity(DataEntity &&other) {}

    virtual ~DataEntity() {}

    virtual std::type_index typeindex() const { return std::type_index(typeid(DataEntity)); }

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataEntity); }

    virtual bool is_null() const { return !(is_table() | is_light() | is_heavy()); }

    virtual bool is_table() const { return false; };

    virtual bool is_light() const { return false; };

    virtual bool is_heavy() const { return false; };

    virtual std::shared_ptr<DataEntity> copy() const { return nullptr; };

    virtual std::shared_ptr<DataEntity> move() { return nullptr; };

    DataEntityTable &as_table();

    DataEntityTable const &as_table() const;

    DataEntityLight &as_light();

    DataEntityLight const &as_light() const;

    DataEntityHeavy &as_heavy();

    DataEntityHeavy const &as_heavy() const;

    template<typename U, typename ...Args> U &as(Args &&...args);

    template<typename U, typename ...Args> U const &as(Args &&...args) const;

    template<typename U> bool equal(U const &u) const;


};

struct DataEntityLight : public DataEntity
{
    SP_OBJECT_HEAD(DataEntityLight, DataEntity);
public:
    template<typename ...Args>
    DataEntityLight(Args &&...args) : m_data_(std::forward<Args>(args)...) {}

    DataEntityLight(DataEntityLight const &other) : m_data_(other.m_data_) {}

    DataEntityLight(DataEntityLight &&other) : m_data_(other.m_data_) {}

    virtual ~DataEntityLight() {}

    virtual bool is_light() const { return true; }

    virtual std::ostream &print(std::ostream &os, int indent) const { return m_data_.print(os, indent); };

    void swap(DataEntityLight &other) { m_data_.swap(other.m_data_); }

    virtual std::shared_ptr<DataEntity> copy() const { return std::make_shared<this_type>(*this); }

    virtual std::shared_ptr<DataEntity> move()
    {
        auto res = std::make_shared<this_type>();
        res->swap(*this);
        return std::dynamic_pointer_cast<DataEntity>(res);
    }

    DataEntityLight &operator=(const DataEntityLight &rhs)
    {
        DataEntityLight(rhs).swap(*this);
        return *this;
    }

    // move assignment
    DataEntityLight &operator=(DataEntityLight &&rhs)
    {
        rhs.swap(*this);
        DataEntityLight().swap(rhs);
        return *this;
    }


    template<typename U>
    DataEntityLight &operator=(U const &v)
    {
        m_data_ = v;
        return *this;
    };


    template<typename U> U as() { return m_data_.as<U>(); };

    template<typename U> U const &as() const { return m_data_.as<U>(); };

    toolbox::Any &any() { return m_data_; }

    toolbox::Any const &any() const { return m_data_; }

private:
    toolbox::Any m_data_;
};

struct DataEntityHeavy : public DataEntity
{
    SP_OBJECT_HEAD(DataEntityHeavy, DataEntity);
public:

    DataEntityHeavy() {}

    virtual ~DataEntityHeavy() {}

    virtual bool is_heavy() const { return true; };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; };


};

template<typename U, typename ...Args> U &
DataEntity::as(Args &&...args) { return as_light().template as<U>(std::forward<Args>(args)...); }

template<typename U, typename ...Args> U const &
DataEntity::as(Args &&...args) const { return as_light().template as<U>(std::forward<Args>(args)...); }

template<typename U> bool
DataEntity::equal(U const &u) const { return as_light().equal(u); }

template<typename U> std::shared_ptr<DataEntity>
create_data_entity(U const &v, ENABLE_IF((std::is_arithmetic<U>::value)))
{
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLight>(v));
};

template<typename U> std::shared_ptr<DataEntity>
create_data_entity(U const &v, ENABLE_IF((!std::is_arithmetic<U>::value)))
{
    UNIMPLEMENTED;
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityHeavy>());
};


}}
#endif //SIMPLA_DATAENTITY_H
