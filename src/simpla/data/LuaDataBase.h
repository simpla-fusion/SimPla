//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/toolbox/LuaObject.h>
#include "DataBase.h"

namespace simpla { namespace data
{

class LuaDataBase : public DataBase
{
public:
    LuaDataBase();

    virtual  ~LuaDataBase();

    virtual std::string name() const { return ""; };

    virtual std::ostream &print(std::ostream &os, int indent) const { return toolbox::LuaObject::print(os, indent); };

    virtual void insert(std::string const &key, std::shared_ptr<DataBase> const &v) { UNIMPLEMENTED; };


    virtual std::shared_ptr<DataBase> create() const
    {
        return std::dynamic_pointer_cast<DataBase>(std::make_shared<LuaDataBase>());
    };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(LuaDataBase) || DataBase::is_a(t_id);
    }

    virtual bool is_table() const;

    virtual bool empty() const;

    virtual bool is_null() const;

    virtual bool has(std::string const &key) const;

//    virtual void set(std::string const &key, std::shared_ptr<DataBase> const &v);

    virtual DataBase &get(std::string const &key);

    virtual DataBase &at(std::string const &key);

    virtual DataBase const &at(std::string const &key) const;

    virtual void foreach(std::function<void(std::string const &key, DataBase const &)> const &) const
    {
        UNIMPLEMENTED;
    };

    virtual void foreach(std::function<void(std::string const &key, DataBase &)> const &)
    {
        UNIMPLEMENTED;
    };

    virtual void set(DataEntity const &other) { UNIMPLEMENTED; };

    virtual void set(DataEntity &&other) { UNIMPLEMENTED; };

    virtual DataEntity &get() { return m_value_; };

    virtual DataEntity const &get() const { return m_value_; };

    LuaDataEntity m_value_;
};

}}//namespace simpla { namespace toolbox {
#endif //SIMPLA_LUADATABASE_H

