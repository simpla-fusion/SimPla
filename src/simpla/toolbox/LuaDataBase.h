//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_LUADATABASE_H
#define SIMPLA_LUADATABASE_H

#include "LuaObject.h"
#include "DataBase.h"

namespace simpla { namespace toolbox
{
class LuaDataEntity : public LuaObject, public DataEntity
{
public:
    LuaDataEntity() {}

    virtual ~LuaDataEntity() {}

    bool empty() const { return LuaObject::empty(); }

    bool is_null() const { return LuaObject::is_null(); }

    void swap(LuaDataEntity &other) { LuaObject::swap(other); }


};

class LuaDataBase : public DataBase
{
public:
    LuaDataBase();

    virtual  ~LuaDataBase();

    void parse_file(std::string const &filename);

    void parse_string(std::string const &str);

    virtual LuaDataEntity const &value() const { return *std::dynamic_pointer_cast<LuaDataEntity const>(m_value_); };

    virtual LuaDataEntity &value() { return *std::dynamic_pointer_cast<LuaDataEntity>(m_value_); };

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

    virtual std::shared_ptr<DataBase> get(std::string const &key);

    virtual std::shared_ptr<DataBase> at(std::string const &key);

    virtual std::shared_ptr<DataBase> at(std::string const &key) const;

};

}}//namespace simpla { namespace toolbox {
#endif //SIMPLA_LUADATABASE_H
