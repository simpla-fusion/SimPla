//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/toolbox/LuaObject.h>
#include "DataTable.h"

namespace simpla { namespace data
{
/** @ingroup data */
/**
 * @brief
 */
class LuaDataBase : public DataTable
{
public:
    LuaDataBase();

    virtual  ~LuaDataBase();

    virtual std::string name() const { return ""; };

    virtual std::ostream &Print(std::ostream &os, int indent) const { return toolbox::LuaObject::Print(os, indent); };

    virtual void insert(std::string const &key, std::shared_ptr<DataTable> const &v) { UNIMPLEMENTED; };


    virtual std::shared_ptr<DataTable> create() const
    {
        return std::dynamic_pointer_cast<DataTable>(std::make_shared<LuaDataBase>());
    };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(LuaDataBase) || DataTable::is_a(t_id);
    }

    virtual bool isTable() const;

    virtual bool empty() const;

    virtual bool isNull() const;

    virtual bool has(std::string const &key) const;

//    virtual void SetValue(std::string const &key, std::shared_ptr<DataTable> const &v);


    virtual DataTable &at(std::string const &key);

    virtual DataTable const &at(std::string const &key) const;

    virtual void foreach(std::function<void(std::string const &key, DataTable const &)> const &) const
    {
        UNIMPLEMENTED;
    };

    virtual void foreach(std::function<void(std::string const &key, DataTable &)> const &)
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

