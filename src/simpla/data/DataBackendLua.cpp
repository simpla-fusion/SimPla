//
// Created by salmon on 17-3-2.
//
#include "DataBackendLua.h"
#include <simpla/toolbox/LuaObject.h>
#include "DataArray.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "DataTraits.h"
namespace simpla {
namespace data {
template <typename U>
struct DataEntityLua;
template <typename U>
struct DataArrayLua;

struct DataBackendLua::pimpl_s {
    toolbox::LuaObject m_lua_obj_;
};
std::shared_ptr<DataEntity> make_data_entity_lua(toolbox::LuaObject const& lobj);

template <typename U>
struct DataEntityLua : public DataEntityWrapper<U> {
    SP_OBJECT_HEAD(DataEntityLua<U>, DataEntityWrapper<U>);

   public:
    DataEntityLua(DataEntityLua<U> const& other) : m_obj_(other.m_obj_){};
    DataEntityLua(toolbox::LuaObject const& v) : m_obj_(v){};
    DataEntityLua(toolbox::LuaObject&& v) : m_obj_(v){};
    virtual ~DataEntityLua(){};

    virtual bool equal(U const& v) const { return m_obj_.as<U>() == v; };
    virtual U value() const { return m_obj_.as<U>(); };

   private:
    toolbox::LuaObject m_obj_;
};

template <typename U>
struct DataArrayLua : public DataArrayWrapper<U> {
    SP_OBJECT_HEAD(DataArrayLua<U>, DataArrayWrapper<U>);

   public:
    DataArrayLua(DataArrayLua<U> const& other) : m_obj_(other.m_obj_){};
    DataArrayLua(toolbox::LuaObject const& v) : m_obj_(v){};
    DataArrayLua(toolbox::LuaObject&& v) : m_obj_(v){};
    virtual ~DataArrayLua(){};

    virtual bool equal(U const& v) const { return m_obj_.as<U>() == v; };
    virtual U value() const { return m_obj_.as<U>(); };

    virtual size_type count() const { return m_obj_.size(); };
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const { return make_data_entity_lua(m_obj_.get(idx)); }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) { return false; }
    virtual bool Add(std::shared_ptr<DataEntity> const&) { return false; }
    virtual int Delete(size_type idx) { return 0; }

   private:
    toolbox::LuaObject m_obj_;
};
std::shared_ptr<DataEntity> make_data_array_lua(toolbox::LuaObject const& lobj) {
    ASSERT(lobj.is_list());
    auto p = lobj[0];
    if (p.is_floating_point()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayLua<double>>(lobj));
    } else if (p.is_integer()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayLua<int>>(lobj));
    } else if (p.is_string()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataArrayLua<std::string>>(lobj));
    }
    return std::shared_ptr<DataEntity>(nullptr);
}
std::shared_ptr<DataEntity> make_data_entity_lua(toolbox::LuaObject const& lobj) {
    if (lobj.is_list()) {
        return make_data_array_lua(lobj);
    } else if (lobj.is_table()) {
        auto p = std::make_unique<DataBackendLua>();
        p->m_pimpl_->m_lua_obj_ = lobj;
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(std::move(p)));
    } else if (lobj.is_boolean()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua<bool>>(lobj));
    } else if (lobj.is_floating_point()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua<double>>(lobj));
    } else if (lobj.is_integer()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua<int>>(lobj));
    } else if (lobj.is_string()) {
        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua<std::string>>(lobj));
    } else {
        RUNTIME_ERROR << "illegal data type of Lua :" << lobj.get_typename() << std::endl;
    }
}

DataBackendLua::DataBackendLua() : m_pimpl_(new pimpl_s) { m_pimpl_->m_lua_obj_.init(); }

DataBackendLua::DataBackendLua(std::string const& url, std::string const& status) : DataBackendLua() {
    m_pimpl_->m_lua_obj_.parse_file(url, status);
}
DataBackendLua::DataBackendLua(DataBackendLua const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_lua_obj_ = other.m_pimpl_->m_lua_obj_;
};
DataBackendLua::~DataBackendLua() {}
std::ostream& DataBackendLua::Print(std::ostream& os, int indent) const {
    return m_pimpl_->m_lua_obj_.Print(os, indent);
}

bool DataBackendLua::IsNull() const { return m_pimpl_->m_lua_obj_.is_null(); }
void DataBackendLua::Flush() {}
void DataBackendLua::Finalize() {}
void DataBackendLua::Initialize() {}
std::unique_ptr<DataBackend> DataBackendLua::Copy() const { return std::make_unique<DataBackendLua>(*this); }

std::shared_ptr<DataEntity> DataBackendLua::Get(std::string const& key) const {
    return make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
};
std::shared_ptr<DataEntity> DataBackendLua::Get(id_type key) const {}

// template <typename K>
// void push_data_to_lua(toolbox::LuaObject& lobj, K const& key, DataArray const& v);
// template <typename K>
// void push_data_to_lua(toolbox::LuaObject& lobj, K const& key, DataTable const& v);
// template <typename K>
// void push_data_to_lua(toolbox::LuaObject& lobj, K const& key, DataEntity const& v);

void add_data_to_lua(toolbox::LuaObject& lobj, DataEntity const& v);

void add_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v);

void set_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v);

void set_data_to_lua(toolbox::LuaObject& lobj, int key, DataEntity const& v) {
    if (key == lobj.size()) {
        add_data_to_lua(lobj, v);
        return;
    } else if (key == lobj.size()) {
        RUNTIME_ERROR << "Out of array boundary" << std::endl;
    }
    toolbox::LuaObject b = lobj.get(key);

    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        for (int s = 0, se = static_cast<int>(v.Count()); s < se; ++s) { set_data_to_lua(b, s, *db.Get(s)); }
    } else if (v.type() == typeid(bool)) {
        lobj.set(key, v.as<bool>());
    } else if (v.type() == typeid(int)) {
        lobj.set(key, v.as<int>());
    } else if (v.type() == typeid(double)) {
        lobj.set(key, v.as<double>());
    } else if (v.type() == typeid(std::string)) {
        lobj.set(key, v.as<std::string>());
    } else {
        RUNTIME_ERROR << "illegal data type for Lua :" << v.type().name() << std::endl;
    }
}
void set_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v) {
    ASSERT(lobj.is_table() || lobj.is_global());

    if (!lobj.has(key)) {
        add_data_to_lua(lobj, key, v);
        return;
    }
    auto b = lobj.get(key);

    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        //         if (b.is_nil()) { b = lobj.new_table(key, 0, static_cast<int>(db.Count())); }
        for (int s = 0, se = static_cast<int>(v.Count()); s < se; ++s) {
            if (s < b.size()) {
                set_data_to_lua(b, s + 1, *db.Get(s));
            } else {
                add_data_to_lua(b, *db.Get(s));
            }
        }
    } else if (v.type() == typeid(bool)) {
        lobj.set(key, v.as<bool>());
    } else if (v.type() == typeid(int)) {
        lobj.set(key, v.as<int>());
    } else if (v.type() == typeid(double)) {
        lobj.set(key, v.as<double>());
    } else if (v.type() == typeid(std::string)) {
        lobj.set(key, v.as<std::string>());
    } else {
        RUNTIME_ERROR << "illegal data type for Lua :" << v.type().name() << std::endl;
    }
}
void add_data_to_lua(toolbox::LuaObject& lobj, DataEntity const& v) {
    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        auto b = lobj.new_table("", 0, db.Count());
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        auto b = lobj.new_table("", db.Count(), 0);
        for (int s = 0, se = static_cast<int>(v.Count()); s < se; ++s) { add_data_to_lua(b, *db.Get(s)); }
    } else if (v.type() == typeid(bool)) {
        lobj.add(v.as<bool>());
    } else if (v.type() == typeid(int)) {
        lobj.add(v.as<int>());
    } else if (v.type() == typeid(double)) {
        lobj.add(v.as<double>());
    } else if (v.type() == typeid(std::string)) {
        lobj.add(v.as<std::string>());
    } else {
        RUNTIME_ERROR << "illegal data type for Lua :" << v.type().name() << std::endl;
    }
}

void add_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v) {
    ASSERT(lobj.is_table() || lobj.is_global());

    if (lobj.has(key)) {
        set_data_to_lua(lobj, key, v);
        return;
    }
    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        auto b = lobj.new_table(key, 0, db.Count());
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        auto b = lobj.new_table(key, db.Count(), 0);
        for (int s = 0, se = static_cast<int>(v.Count()); s < se; ++s) { add_data_to_lua(b, *db.Get(s)); }
    } else if (v.type() == typeid(bool)) {
        lobj.set(key, v.as<bool>());
    } else if (v.type() == typeid(int)) {
        lobj.set(key, v.as<int>());
    } else if (v.type() == typeid(double)) {
        lobj.set(key, v.as<double>());
    } else if (v.type() == typeid(std::string)) {
        lobj.set(key, v.as<std::string>());
    } else {
        RUNTIME_ERROR << "illegal data type for Lua :" << v.type().name() << std::endl;
    }
}
bool DataBackendLua::Set(std::string const& key, std::shared_ptr<DataEntity> const& v) {
    set_data_to_lua(m_pimpl_->m_lua_obj_, key, *v);
    return true;
}
bool DataBackendLua::Set(id_type key, std::shared_ptr<DataEntity> const&) {}

bool DataBackendLua::Add(std::string const& key, std::shared_ptr<DataEntity> const& v) { return false; }
bool DataBackendLua::Add(id_type key, std::shared_ptr<DataEntity> const&) {}

size_type DataBackendLua::Delete(std::string const& key) { return 0; }
size_type DataBackendLua::Delete(id_type key) {}
void DataBackendLua::DeleteAll() {}
size_type DataBackendLua::Count(std::string const& uri) const { return 0; }
size_type DataBackendLua::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    return m_pimpl_->m_lua_obj_.accept([&](toolbox::LuaObject const& k, toolbox::LuaObject const& v) {
        f(k.as<std::string>(), make_data_entity_lua(v));
    });
}
size_type DataBackendLua::Accept(std::function<void(id_type, std::shared_ptr<DataEntity>)> const&) const {};

// std::shared_ptr<DataEntity> DataBackendLua::Get(std::string const& url) {
//    auto obj = m_pimpl_->m_lua_obj_.get(url);
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto backend = std::make_shared<DataBackendLua>();
//        backend->m_pimpl_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(backend));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << ":" << obj.get_typename() << std::endl;
//    }
//};
// std::shared_ptr<DataEntity> DataBackendLua::Get(std::string const& url) const {
//    auto obj = m_pimpl_->m_lua_obj_.get(url);
//    ASSERT(!obj.empty());
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto backend = std::make_shared<DataBackendLua>();
//        backend->m_pimpl_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(backend));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << std::endl;
//    }
//}

}  // namespace data{
}  // namespace simpla