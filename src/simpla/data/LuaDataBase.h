//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include <simpla/toolbox/LuaObject.h>
#include "DataEntity.h"
#include "DataBase.h"

namespace simpla { namespace data
{
class LuaDataEntity : public toolbox::LuaObject, public DataEntity
{
public:
    LuaDataEntity() {}

    virtual ~LuaDataEntity() {}

    bool empty() const { return toolbox::LuaObject::empty(); }

    bool is_null() const { return toolbox::LuaObject::is_null(); }

    void swap(LuaDataEntity &other) { toolbox::LuaObject::swap(other); }


};

class LuaDataBase : public DataBase
{
public:
    LuaDataBase();

    virtual  ~LuaDataBase();

    void parse_file(std::string const &filename);

    void parse_string(std::string const &str);

//    virtual LuaDataEntity const &value() const;
//
//    virtual LuaDataEntity &value();

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

    virtual void set(std::string const &key, std::shared_ptr<DataBase> const &v);

    virtual DataBase &get(std::string const &key);

    virtual DataBase &at(std::string const &key);

    virtual DataBase const &at(std::string const &key) const;

};

}}//namespace simpla { namespace toolbox {
#endif //SIMPLA_LUADATABASE_H
