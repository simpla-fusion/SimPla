//
// Created by salmon on 17-3-2.
//

#include "DataTableLua.h"
namespace simpla {
namespace data {

DataEntityLua::DataEntityLua(toolbox::LuaObject const& u) : m_lua_obj_(u){};
DataEntityLua::DataEntityLua(toolbox::LuaObject&& u) : m_lua_obj_(u){};
DataEntityLua::~DataEntityLua() {}
std::ostream& DataEntityLua::Print(std::ostream& os, int indent) const { return m_lua_obj_.Print(os, indent); };
bool DataEntityLua::isLight() const { return true; };

std::shared_ptr<DataEntity> DataEntityLua::Copy() const {
    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua>(m_lua_obj_));
};

DataTableLua::DataTableLua() {}
DataTableLua::DataTableLua(DataTableLua const& other) : m_lua_obj_(other.m_lua_obj_) {}
DataTableLua::DataTableLua(toolbox::LuaObject const& other) : m_lua_obj_(other) {}
DataTableLua::DataTableLua(toolbox::LuaObject&& other) : m_lua_obj_(other) {}
DataTableLua::~DataTableLua() {}
void DataTableLua::ParseFile(std::string const& str) {
    m_lua_obj_.init();
    m_lua_obj_.parse_file(str);
}
void DataTableLua::Parse(std::string const& str) {
    m_lua_obj_.init();
    m_lua_obj_.parse_string(str);
}
std::ostream& DataTableLua::Print(std::ostream& os, int indent) const { return m_lua_obj_.Print(os, indent); }
bool DataTableLua::isTable() const { return true; };
bool DataTableLua::empty() const { return m_lua_obj_.empty(); }
bool DataTableLua::has(std::string const& key) const { return m_lua_obj_.has(key); }

void DataTableLua::Set(std::string const& key, std::shared_ptr<DataEntity> const& v) { UNIMPLEMENTED; }

std::shared_ptr<DataEntityLua> DataTableLua::GetLua(std::string const& url) {
    return std::make_shared<DataEntityLua>(m_lua_obj_.get(url));
};
std::shared_ptr<DataEntityLua> DataTableLua::GetLua(std::string const& url) const {
    return std::make_shared<DataEntityLua>(m_lua_obj_.get(url));
};
std::shared_ptr<DataEntity> DataTableLua::Get(std::string const& url) {
    return std::dynamic_pointer_cast<DataEntity>(GetLua(url));
};
std::shared_ptr<DataEntity> DataTableLua::Get(std::string const& url) const {
    return std::dynamic_pointer_cast<DataEntity>(GetLua(url));
}
std::shared_ptr<DataTable> DataTableLua::CreateTable(std::string const& url) {
    return std::dynamic_pointer_cast<DataTable>(std::make_shared<DataTableLua>(m_lua_obj_.new_table(url)));
};
}  // namespace data{
}  // namespace simpla