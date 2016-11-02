//
// Created by salmon on 16-10-28.
//
#include "LuaDataBase.h"

namespace simpla { namespace data
{
LuaDataBase::LuaDataBase()
{
    m_value_ = std::shared_ptr<DataEntity>((new LuaDataEntity));
    std::dynamic_pointer_cast<LuaDataEntity>(m_value_)->init();
};

LuaDataBase::~LuaDataBase() {};


void LuaDataBase::parse_file(std::string const &filename) { value().parse_file(filename); };

void LuaDataBase::parse_string(std::string const &str) { value().parse_file(str); };

//bool LuaDataBase::is_a(std::type_info const &t_id) const
//{
//    return t_id == typeid(LuaDataBase) || DataBase::is_a(t_id);
//}

bool LuaDataBase::is_table() const { return value().is_table() || DataBase::is_table(); };

bool LuaDataBase::empty() const { return value().empty() || DataBase::empty(); };

bool LuaDataBase::is_null() const { return value().is_null() || DataBase::is_null(); };

bool LuaDataBase::has(std::string const &key) const { return value().has(key) || DataBase::has(key); };

std::shared_ptr<DataBase> LuaDataBase::get(std::string const &key)
{
    auto res = std::make_shared<LuaDataBase>();

//    res->m_value_().swap(this->m_value_().LuaObject::get(key));

    return (res->value().is_nil()) ? DataBase::get(key) : std::dynamic_pointer_cast<DataBase>(res);

};

std::shared_ptr<DataBase> LuaDataBase::at(std::string const &key)
{
    auto res = std::make_shared<LuaDataBase>();

//    res->m_value_().swap(this->m_value_().get(key));

    return (res->value().is_nil()) ? DataBase::at(key) : std::dynamic_pointer_cast<DataBase>(res);

};

std::shared_ptr<DataBase> LuaDataBase::at(std::string const &key) const
{
    auto res = std::make_shared<LuaDataBase>();

//    res->m_value_().swap(this->m_value_()[key]);

    return (res->value().is_nil()) ? DataBase::at(key) : std::dynamic_pointer_cast<DataBase>(res);

};

}}//namespace simpla { namespace toolbox