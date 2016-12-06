//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/toolbox/LuaObject.h>
#include "DataEntityTable.h"

namespace simpla { namespace data
{

class LuaDataBase : public DataEntityTable
{
public:
    LuaDataBase();

    virtual  ~LuaDataBase();

    virtual std::string name() const { return ""; };

    virtual std::ostream &print(std::ostream &os, int indent) const { return toolbox::LuaObject::print(os, indent); };

    virtual void insert(std::string const &key, std::shared_ptr<DataEntityTable> const &v) { UNIMPLEMENTED; };


    virtual std::shared_ptr<DataEntityTable> create() const
    {
        return std::dynamic_pointer_cast<DataEntityTable>(std::make_shared<LuaDataBase>());
    };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(LuaDataBase) || DataEntityTable::is_a(t_id);
    }

    virtual bool is_table() const;

    virtual bool empty() const;

    virtual bool is_null() const;

    virtual bool has(std::string const &key) const;

//    virtual void set_value(std::string const &key, std::shared_ptr<DataEntityTable> const &v);

    virtual DataEntityTable &get(std::string const &key);

    virtual DataEntityTable &at(std::string const &key);

    virtual DataEntityTable const &at(std::string const &key) const;

    virtual void foreach(std::function<void(std::string const &key, DataEntityTable const &)> const &) const
    {
        UNIMPLEMENTED;
    };

    virtual void foreach(std::function<void(std::string const &key, DataEntityTable &)> const &)
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

