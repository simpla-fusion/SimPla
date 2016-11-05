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
#include <simpla/toolbox/Printable.h>


namespace simpla { namespace data
{
class DataEntity;

class DataBase : public toolbox::Printable
{
public:

    DataBase() {};

    DataBase(DataBase const &) = delete;

    DataBase(DataBase &&) = delete;

    virtual  ~DataBase() {};

    virtual std::string  name() const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual DataEntity const &value() const =0;

    virtual DataEntity &value() =0;

    virtual std::shared_ptr<DataBase> create() const =0;

    virtual bool is_a(std::type_info const &t_id) const =0;

    virtual bool is_table() const =0;

    virtual bool empty() const =0;

    virtual bool is_null() const =0;

    virtual bool has(std::string const &key) const =0;

    virtual void insert(std::string const &key, std::shared_ptr<DataBase> const &v) =0;

    virtual std::shared_ptr<DataBase> find(std::string const &key)  =0;

    virtual std::shared_ptr<const DataBase> find(std::string const &key) const =0;

    virtual DataBase &at(std::string const &key)  =0;

    virtual DataBase const &at(std::string const &key) const =0;

    virtual DataBase &get(std::string const &key) =0;

    DataBase &operator[](std::string const &key) { return get(key); };


    virtual void foreach(std::function<void(std::string const &key, DataBase const &)> const &) const =0;

    virtual void foreach(std::function<void(std::string const &key, DataBase &)> const &)=0;

};


}}//namespace simpla{namespace toolbox{

#endif //SIMPLA_DICT_H
