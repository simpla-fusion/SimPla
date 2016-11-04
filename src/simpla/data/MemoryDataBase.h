//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_MEMORYDATABASE_H
#define SIMPLA_MEMORYDATABASE_H

#include "DataEntity.h"
#include "DataBase.h"

namespace simpla { namespace data
{

class MemoryDataBase : public DataBase
{
    MemoryDataBase();

    virtual  ~MemoryDataBase();

    MemoryDataBase(MemoryDataBase const &) = delete;

    MemoryDataBase(MemoryDataBase &&) = delete;

    virtual DataEntity const &value() const { return *m_value_; };

    virtual DataEntity &value() { return *m_value_; };

    virtual std::shared_ptr<DataBase> create() const { return std::make_shared<MemoryDataBase>(); };

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataBase); }

    virtual bool is_table() const { return (m_value_ == nullptr) && !m_table_.empty(); };

    virtual bool empty() const { return m_value_ == nullptr || m_table_.empty(); };

    virtual bool is_null() const { return empty(); };

    virtual bool has(std::string const &key) const { return m_table_.find(key) != m_table_.end(); };

    bool check(std::string const &key)
    {
        return has(key) && static_cast<DataEntityLight &>(at(key).value()).template as<bool>();
    }

    virtual void insert(std::string const &key, std::shared_ptr<MemoryDataBase> const &v)
    {
        m_table_.emplace(key, v);
    };;

    virtual MemoryDataBase &get(std::string const &key) { return *m_table_[key]; };

    virtual MemoryDataBase &at(std::string const &key) { return *m_table_.at(key); };

    virtual MemoryDataBase const &at(std::string const &key) const { return *m_table_.at(key); };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void foreach(std::function<void(std::string const &key, DataBase const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, DataBase &)> const &fun);

    virtual void foreach(std::function<void(std::string const &key, MemoryDataBase const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, MemoryDataBase &)> const &fun);


protected:

    std::map<std::string, std::shared_ptr<MemoryDataBase>> m_table_;
    std::shared_ptr<DataEntity> m_value_;
};
}}
#endif //SIMPLA_MEMORYDATABASE_H
