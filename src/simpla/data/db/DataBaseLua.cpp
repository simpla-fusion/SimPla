//
// Created by salmon on 17-3-2.
//
#include "DataBaseLua.h"
#include "../DataArray.h"
#include "../DataEntity.h"
#include "../DataTable.h"
#include "../DataTraits.h"
#include "LuaObject.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseLua, lua);

struct DataBaseLua::pimpl_s {
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
DataBaseLua::DataBaseLua() : m_pimpl_(new pimpl_s) { m_pimpl_->m_lua_obj_.init(); }
DataBaseLua::~DataBaseLua() { delete m_pimpl_; }

// std::ostream& DataBaseLua::Print(std::ostream& os, int indent) const { return m_pimpl_->m_lua_obj_.Print(os, indent);
// }

int DataBaseLua::Flush() { return SP_SUCCESS; }
// void DataBaseLua::Parser(std::string const& str) { m_pimpl_->m_lua_obj_.parse_string(str); }

int DataBaseLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    m_pimpl_->m_lua_obj_.parse_file(path);
    return SP_SUCCESS;
}
int DataBaseLua::Disconnect() { return SP_SUCCESS; }

template <typename U>
std::shared_ptr<DataEntity> DataBaseLua::pimpl_s::make_data_array_lua(LuaObject const& lobj) {
    auto res = DataArrayT<U>::New();
    for (auto const& item : lobj) { res->Add(item.second.as<U>()); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
std::shared_ptr<DataEntity> DataBaseLua::pimpl_s::make_data_entity_lua(LuaObject const& lobj) {
    std::shared_ptr<DataEntity> res = nullptr;

    if (lobj.is_table()) {
        auto p = DataBaseLua::New();
        p->m_pimpl_->m_lua_obj_ = lobj;
        res = DataTable::New(p);
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
std::shared_ptr<DataEntity> DataBaseLua::Get(std::string const& key) const {
    return m_pimpl_->make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
};

// int DataBaseLua::pack_s::set_data_to_lua(LuaObject& lobj, int key, std::shared_ptr<DataEntity> const& v,
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
//        for (int s = 0, se = static_cast<int>(db->size()); s < se; ++s) { set_data_to_lua(b, s, db->GetPatch(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(id_type)) {
//        lobj.set(key, static_cast<int>(DataCastTraits<id_type>::GetPatch(v)));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(float)) {
//        lobj.set(key, DataCastTraits<float>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::GetPatch(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
// int DataBaseLua::pack_s::set_data_to_lua(LuaObject& lobj, std::string const& key,
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
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.GetPatch(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::GetPatch(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
// int DataBaseLua::pack_s::add_data_to_lua(LuaObject& lobj, std::shared_ptr<DataEntity> const& v) {
//    if (v->isTable()) {
//        auto const& db = dynamic_cast<DataTable const&>(*v);
//        auto b = lobj.new_table("", 0, db.size());
//        db.Foreach([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, p); });
//    } else if (v->isArray()) {
//        auto const& db = dynamic_cast<DataArray const&>(*v);
//        auto b = lobj.new_table("", db.size(), 0);
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.GetPatch(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.add(DataCastTraits<bool>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.add(DataCastTraits<int>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.add(DataCastTraits<double>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.add(DataCastTraits<std::string>::GetPatch(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
//
// int DataBaseLua::pack_s::add_data_to_lua(LuaObject& lobj, std::string const& key,
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
//        for (int s = 0, se = static_cast<int>(db.size()); s < se; ++s) { add_data_to_lua(b, db.GetPatch(s)); }
//    } else if (v->value_type_info() == typeid(bool)) {
//        lobj.set(key, DataCastTraits<bool>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(int)) {
//        lobj.set(key, DataCastTraits<int>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(double)) {
//        lobj.set(key, DataCastTraits<double>::GetPatch(v));
//    } else if (v->value_type_info() == typeid(std::string)) {
//        lobj.set(key, DataCastTraits<std::string>::GetPatch(v));
//    } else {
//        RUNTIME_ERROR << "illegal data type for Lua :" << v->value_type_info().name() << std::endl;
//    }
//    return 1;
//}
int DataBaseLua::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
    if (v == nullptr) { m_pimpl_->m_lua_obj_.parse_string(uri); }
    return 1;
}

int DataBaseLua::Add(std::string const& key, const std::shared_ptr<DataEntity>& v) {
    UNIMPLEMENTED;
    return 0;
}
int DataBaseLua::Delete(std::string const& key) {
    UNIMPLEMENTED;
    return 0;
}
bool DataBaseLua::isNull() const { return m_pimpl_->m_lua_obj_.is_nil(); }

int DataBaseLua::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    int counter = 0;
    if (m_pimpl_->m_lua_obj_.is_global()) {
        UNSUPPORTED;
    } else {
        for (auto const& item : m_pimpl_->m_lua_obj_) {
            if (item.first.is_string()) {
                counter += f(item.first.as<std::string>(), this->Get(item.first.as<std::string>()));
            }
        };
    }
    return counter;
}

// std::shared_ptr<DataEntity> DataBaseLua::Serialize(std::string const& url) {
//    auto obj = m_pack_->m_lua_obj_.get(url);
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto database = std::make_shared<DataBaseLua>();
//        database->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(database));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << ":" << obj.get_typename() << std::endl;
//    }
//};
// std::shared_ptr<DataEntity> DataBaseLua::Serialize(std::string const& url) const {
//    auto obj = m_pack_->m_lua_obj_.get(url);
//    ASSERT(!obj.empty());
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto database = std::make_shared<DataBaseLua>();
//        database->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(database));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << std::endl;
//    }
//}

}  // namespace data{
}  // namespace simpla