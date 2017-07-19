//
// Created by salmon on 17-3-2.
//
#include "DataBackendLua.h"
#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
#include "../DataTraits.h"
#include "LuaObject.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBackendLua);

struct DataBackendLua::pimpl_s {
    LuaObject m_lua_obj_;

    //    static int add_data_to_lua(LuaObject& lobj, std::shared_ptr<DataEntity> const& v);
    //    static int add_data_to_lua(LuaObject& lobj, std::string const& key, std::shared_ptr<DataEntity>
    //    const& v);
    //    static int set_data_to_lua(LuaObject& lobj, std::string const& key, std::shared_ptr<DataEntity>
    //    const& v,
    //                               bool overwrite = true);
    //    static int set_data_to_lua(LuaObject& lobj, int key, std::shared_ptr<DataEntity> const& v,
    //                               bool overwrite = true);

    template <typename U>
    std::shared_ptr<DataEntity> make_data_array_lua(LuaObject const& lobj);
    std::shared_ptr<DataEntity> make_data_entity_lua(LuaObject const& lobj);
};
DataBackendLua::DataBackendLua() : m_pimpl_(new pimpl_s) { m_pimpl_->m_lua_obj_.init(); }
DataBackendLua::DataBackendLua(DataBackendLua const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_lua_obj_ = other.m_pimpl_->m_lua_obj_;
};
DataBackendLua::~DataBackendLua() {}
std::ostream& DataBackendLua::Print(std::ostream& os, int indent) const {
    return m_pimpl_->m_lua_obj_.Print(os, indent);
}

bool DataBackendLua::isNull() const { return m_pimpl_->m_lua_obj_.is_null(); }
void DataBackendLua::Flush() {}
void DataBackendLua::Parser(std::string const& str) { m_pimpl_->m_lua_obj_.parse_string(str); }

void DataBackendLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                             std::string const& fragment) {
    m_pimpl_->m_lua_obj_.parse_file(path);
}
void DataBackendLua::Disconnect() {}
std::shared_ptr<DataBackend> DataBackendLua::Duplicate() const { return std::make_shared<DataBackendLua>(*this); }
std::shared_ptr<DataBackend> DataBackendLua::CreateNew() const { return std::make_shared<DataBackendLua>(); }

template <typename U>
std::shared_ptr<DataEntity> DataBackendLua::pimpl_s::make_data_array_lua(LuaObject const& lobj) {
    auto res = std::make_shared<DataEntityWrapper<U*>>();
    for (auto const& item : lobj) { res->Add(item.second.as<U>()); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
std::shared_ptr<DataEntity> DataBackendLua::pimpl_s::make_data_entity_lua(LuaObject const& lobj) {
    std::shared_ptr<DataEntity> res = nullptr;

    if (lobj.is_table()) {
        auto p = std::make_unique<DataBackendLua>();
        p->m_pimpl_->m_lua_obj_ = lobj;
        res = std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(std::move(p)));
    } else if (lobj.is_array()) {
        auto a = *lobj.begin();
        if (a.second.is_integer()) {
            res = make_data_array_lua<int>(lobj);
        } else if (a.second.is_floating_point()) {
            res = make_data_array_lua<double>(lobj);
        } else if (a.second.is_string()) {
            res = make_data_array_lua<std::string>(lobj);
        }
    } else if (lobj.is_boolean()) {
        res = make_data_entity(lobj.as<bool>());
    } else if (lobj.is_floating_point()) {
        res = make_data_entity<double>(lobj.as<double>());
    } else if (lobj.is_integer()) {
        res = make_data_entity<int>(lobj.as<int>());
    } else if (lobj.is_string()) {
        res = make_data_entity<std::string>(lobj.as<std::string>());
    } else {
        RUNTIME_ERROR << "illegal data type of Lua :" << lobj.get_typename() << std::endl;
    }
    return res;
}
std::shared_ptr<DataEntity> DataBackendLua::Get(std::string const& key) const {
    return m_pimpl_->make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
};
std::shared_ptr<DataEntity> DataBackendLua::Get(int key) const {
    return m_pimpl_->make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
}

// int DataBackendLua::pack_s::set_data_to_lua(LuaObject& lobj, int key, std::shared_ptr<DataEntity> const& v,
//                                             bool overwrite) {
//    if (key == lobj.size()) { return add_data_to_lua(lobj, v); }
//
//    if (v->isTable()) {
//        LuaObject b = lobj.get(key);
//        auto const db = std::dynamic_pointer_cast<DataTable>(v);
//        db->Foreach([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, p); });
//    } else if (v->isArray()) {
//        LuaObject b = lobj.get(key);
//        auto const db = std::dynamic_pointer_cast<DataArray>(v);
//        for (int s = 0, se = static_cast<int>(db->size()); s < se; ++s) { set_data_to_lua(b, s, db->Get(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::Get(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::Get(v));
//    } else if (v->value_type_info() == typeid(id_type)) {
//        lobj.set(key, static_cast<int>(DataCastTraits<id_type>::Get(v)));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::Get(v));
//    } else if (v->value_type_info() == typeid(float)) {
//        lobj.set(key, DataCastTraits<float>::Get(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::Get(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
// int DataBackendLua::pack_s::set_data_to_lua(LuaObject& lobj, std::string const& key,
//                                             std::shared_ptr<DataEntity> const& v, bool overwrite) {
//    ASSERT(lobj.is_table() || lobj.is_global());
//    if (lobj.has(key) && !overwrite) { return 0; }
//
//    if (v->isTable()) {
//        auto const& db = v->cast_as<DataTable>();
//        auto b = lobj.new_table(key, 0, db.size());
//        db.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, p); });
//    } else if (v->isArray()) {
//        auto const& db = v->cast_as<DataArray>();
//        auto b = lobj.new_table(key, db.size(), 0);
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.Get(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::Get(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::Get(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::Get(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::Get(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
// int DataBackendLua::pack_s::add_data_to_lua(LuaObject& lobj, std::shared_ptr<DataEntity> const& v) {
//    if (v->isTable()) {
//        auto const& db = dynamic_cast<DataTable const&>(*v);
//        auto b = lobj.new_table("", 0, db.size());
//        db.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, p); });
//    } else if (v->isArray()) {
//        auto const& db = dynamic_cast<DataArray const&>(*v);
//        auto b = lobj.new_table("", db.size(), 0);
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.Get(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.add(DataCastTraits<bool>::Get(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.add(DataCastTraits<int>::Get(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.add(DataCastTraits<double>::Get(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.add(DataCastTraits<std::string>::Get(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
//
// int DataBackendLua::pack_s::add_data_to_lua(LuaObject& lobj, std::string const& key,
//                                             std::shared_ptr<DataEntity> const& v) {
//    ASSERT(lobj.is_table() || lobj.is_global());
//
//    if (lobj.has(key)) { return set_data_to_lua(lobj, key, v); }
//    if (v->isTable()) {
//        auto const& db = dynamic_cast<DataTable const&>(*v);
//        auto b = lobj.new_table(key, 0, db.size());
//        db.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, p); });
//    } else if (v->isArray()) {
//        auto const& db = dynamic_cast<DataArray const&>(*v);
//        auto b = lobj.new_table(key, db.size(), 0);
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.Get(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::Get(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::Get(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::Get(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::Get(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
void DataBackendLua::Set(std::string const& key, std::shared_ptr<DataEntity> const& v, bool overwrite) {
    UNIMPLEMENTED;
    //    DataBackendLua::pack_s::set_data_to_lua(m_pack_->m_lua_obj_, key, v, overwrite);
}

void DataBackendLua::Add(std::string const& key, std::shared_ptr<DataEntity> const& v) { UNIMPLEMENTED; }
int DataBackendLua::Delete(std::string const& key) {
    UNIMPLEMENTED;
    return 0;
}
size_type DataBackendLua::size() const { return m_pimpl_->m_lua_obj_.size(); }

size_type DataBackendLua::Foreach(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    if (m_pimpl_->m_lua_obj_.is_global()) {
        UNSUPPORTED;
    } else {
        for (auto const& item : m_pimpl_->m_lua_obj_) {
            if (item.first.is_string()) { f(item.first.as<std::string>(), this->Get(item.first.as<std::string>())); }
        };
    }
    return 0;
}

// std::shared_ptr<DataEntity> DataBackendLua::Serialize(std::string const& url) {
//    auto obj = m_pack_->m_lua_obj_.get(url);
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto backend = std::make_shared<DataBackendLua>();
//        backend->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(backend));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << ":" << obj.get_typename() << std::endl;
//    }
//};
// std::shared_ptr<DataEntity> DataBackendLua::Serialize(std::string const& url) const {
//    auto obj = m_pack_->m_lua_obj_.get(url);
//    ASSERT(!obj.empty());
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto backend = std::make_shared<DataBackendLua>();
//        backend->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(backend));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << std::endl;
//    }
//}

}  // namespace data{
}  // namespace simpla