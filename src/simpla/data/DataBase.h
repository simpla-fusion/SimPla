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

class DataBase : public DataEntity
{
public:

    DataBase();

    DataBase(DataBase const &) = delete;

    DataBase(DataBase &&) = delete;

    virtual  ~DataBase();

    virtual std::string name() const { return ""; };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual std::shared_ptr<DataBase> clone() const { return std::make_shared<DataBase>(); };

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataBase); };

    virtual bool is_table() const;

    virtual bool is_null() const;

    virtual bool empty() const;

    virtual bool has(std::string const &key) const;

    virtual void insert(std::string const &key, std::shared_ptr<DataBase> const &v);

    virtual void foreach(std::function<void(std::string const &key, DataBase const &)> const &) const;

    virtual void foreach(std::function<void(std::string const &key, DataBase &)> const &fun);

    virtual bool check(std::string const &key);

    virtual DataBase &get(std::string const &key);

    virtual DataBase &at(std::string const &key);

    virtual DataBase const &at(std::string const &key) const;

    virtual DataBase &add(std::string const &key);

    DataBase &operator[](std::string const &key) { return get(key); };

    DataBase &operator[](char const *key) { return get(key); };

    DataBase const &operator[](char const *&key) const { return at(key); };

    DataBase const &operator[](std::string const &key) const { return at(key); };

    template<typename U>
    DataBase &operator=(U const &v)
    {
        DataEntity::operator=(v);
        return *this;
    };

    template<typename U>
    U const &get(std::string const &key, U const &v) { if (has(key)) { return at(key).as<U>(); } else { return v; }}

    template<typename T> bool equal(T const &v) const { return is_a(typeid(T)) && as<T>() == v; };

    template<typename T> bool operator==(T const &v) const { return equal(v); }

    template<typename T> bool equal(std::string const &k, T const &v) const { return has(k) && at(k).equal(v); };


protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
