//
// Created by salmon on 16-10-6.
//

#ifndef SIMPLA_DICT_H
#define SIMPLA_DICT_H


#include <simpla/SIMPLA_config.h>

#include <typeinfo>
#include <string>
#include <memory>
#include <ostream>
#include <iomanip>
#include <map>
#include "DataEntity.h"

namespace simpla { namespace data
{


class DataBase
{
public:

    DataBase();

    virtual  ~DataBase();

    DataBase(DataBase const &) = delete;

    DataBase(DataBase &&) = delete;

    virtual DataEntity const &value() const { return *m_value_; };

    virtual DataEntity &value() { return *m_value_; };

    virtual std::shared_ptr<DataBase> create() const { return std::make_shared<DataBase>(); };

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataBase); }

    virtual bool is_table() const { return (m_value_ == nullptr) && !m_table_.empty(); };

    virtual bool empty() const { return m_value_ == nullptr || m_table_.empty(); };

    virtual bool is_null() const { return empty(); };

    virtual bool has(std::string const &key) const { return m_table_.find(key) != m_table_.end(); };

    virtual void set(std::string const &key, std::shared_ptr<DataBase> const &v) { get(key) = v; };;

    virtual std::shared_ptr<DataBase> get(std::string const &key) { return m_table_[key]; };

    virtual std::shared_ptr<DataBase> at(std::string const &key) { return m_table_.at(key); };

    virtual std::shared_ptr<DataBase> at(std::string const &key) const { return m_table_.at(key); };

    std::shared_ptr<DataBase> operator[](std::string const &key) { return get(key); };

    template<typename T> T const &
    value_as(std::string const &key) const { return at(key)->value().template as<T>(); }

    template<typename T> T const &
    value_as(std::string const &key, T const &default_value) const
    {
        if (has(key)) { return at(key)->value().template as<T>(); } else { return default_value; }
    }

    bool check(std::string const &key) { return has(key) && at(key)->value().template as<bool>(); }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void foreach(std::function<void(std::string const &, std::shared_ptr<DataBase> const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &, std::shared_ptr<DataBase> &)> const &);

protected:

    std::map<std::string, std::shared_ptr<DataBase> > m_table_;
    std::shared_ptr<DataEntity> m_value_;
};

std::ostream &operator<<(std::ostream &os, DataEntity const &prop);

std::ostream &operator<<(std::ostream &os, DataBase const &prop);


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
