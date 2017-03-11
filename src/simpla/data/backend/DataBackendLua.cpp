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
constexpr char DataBackendLua::scheme_tag[];

template <typename U>
struct DataEntityLua;
template <typename U>
struct DataArrayLua;

struct DataBackendLua::pimpl_s {
    toolbox::LuaObject m_lua_obj_;
    template <typename U>
    static std::shared_ptr<DataEntity> make_data_array_lua(toolbox::LuaObject const& lobj);
    static std::shared_ptr<DataEntity> make_data_entity_lua(toolbox::LuaObject const& lobj);
    static void add_data_to_lua(toolbox::LuaObject& lobj, DataEntity const& v);
    static void add_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v);
    static void set_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v);
    static void set_data_to_lua(toolbox::LuaObject& lobj, int key, DataEntity const& v);
};

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
    virtual std::shared_ptr<DataEntity> Get(size_type idx) const {
        return DataBackendLua::pimpl_s::make_data_entity_lua(m_obj_.get(idx));
    }
    virtual bool Set(size_type idx, std::shared_ptr<DataEntity> const&) { return false; }
    virtual bool Add(std::shared_ptr<DataEntity> const&) { return false; }
    virtual int Delete(size_type idx) { return 0; }

   private:
    toolbox::LuaObject m_obj_;
};

template <typename U>
std::shared_ptr<DataEntity> DataBackendLua::pimpl_s::make_data_array_lua(toolbox::LuaObject const& lobj) {
    ASSERT(lobj.is_list());
    auto res = std::make_shared<DataArrayWrapper<U>>();
    for (auto const& item : lobj) { res->Add(item.second.as<U>()); }
    return std::dynamic_pointer_cast<DataEntity>(res);
}
std::shared_ptr<DataEntity> DataBackendLua::pimpl_s::make_data_entity_lua(toolbox::LuaObject const& lobj) {
    if (lobj.is_list()) {
        auto p = lobj[0];
        if (p.is_floating_point()) {
            return make_data_array_lua<double>(lobj);
        } else if (p.is_integer()) {
            return make_data_array_lua<int>(lobj);
        } else if (p.is_string()) {
            return make_data_array_lua<std::string>(lobj);
        }
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
    } else if (lobj.is_global()) {
        RUNTIME_ERROR << "illegal data type of Lua :" << lobj.get_typename() << std::endl;
    } else {
        RUNTIME_ERROR << "illegal data type of Lua :" << lobj.get_typename() << std::endl;
    }
}

DataBackendLua::DataBackendLua() : m_pimpl_(new pimpl_s) { m_pimpl_->m_lua_obj_.init(); }

DataBackendLua::DataBackendLua(DataBackendLua const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_lua_obj_ = other.m_pimpl_->m_lua_obj_;
};
DataBackendLua::~DataBackendLua() {}

std::ostream& DataBackendLua::Print(std::ostream& os, int indent) const {
    return m_pimpl_->m_lua_obj_.Print(os, indent);
}
std::string DataBackendLua::scheme() const { return scheme_tag; }

bool DataBackendLua::isNull() const { return m_pimpl_->m_lua_obj_.is_null(); }
void DataBackendLua::Flush() {}
void DataBackendLua::Connect(std::string const& path) { m_pimpl_->m_lua_obj_.parse_file(path); }
void DataBackendLua::Disconnect() {}
DataBackend* DataBackendLua::Clone() const { return new DataBackendLua(*this); }
DataBackend* DataBackendLua::Create() const { return new DataBackendLua(); }

std::shared_ptr<DataEntity> DataBackendLua::Get(std::string const& key) const {
    return DataBackendLua::pimpl_s::make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
};
std::shared_ptr<DataEntity> DataBackendLua::Get(id_type key) const {}

void DataBackendLua::pimpl_s::set_data_to_lua(toolbox::LuaObject& lobj, int key, DataEntity const& v) {
    if (key == lobj.size()) {
        add_data_to_lua(lobj, v);
        return;
    }
    //    else if (key > lobj.size()) {
    //        RUNTIME_ERROR << "Out of array boundary" << std::endl;
    //    }

    if (v.isTable()) {
        toolbox::LuaObject b = lobj.get(key);
        auto const& db = dynamic_cast<DataTable const&>(v);
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        toolbox::LuaObject b = lobj.get(key);
        auto const& db = dynamic_cast<DataArray const&>(v);
        for (int s = 0, se = static_cast<int>(v.size()); s < se; ++s) { set_data_to_lua(b, s, *db.Get(s)); }
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
void DataBackendLua::pimpl_s::set_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v) {
    ASSERT(lobj.is_table() || lobj.is_global());

    if (v.isTable()) {
        auto const& db = v.cast_as<DataTable>();
        auto b = lobj.new_table(key, 0, db.size());
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = v.cast_as<DataArray>();
        CHECK(db.size());
        auto b = lobj.new_table(key, db.size(), 0);
        for (int s = 0, se = static_cast<int>(v.size()); s < se; ++s) { add_data_to_lua(b, *db.Get(s)); }
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
void DataBackendLua::pimpl_s::add_data_to_lua(toolbox::LuaObject& lobj, DataEntity const& v) {
    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        auto b = lobj.new_table("", 0, db.size());
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        auto b = lobj.new_table("", db.size(), 0);
        for (int s = 0, se = static_cast<int>(v.size()); s < se; ++s) { add_data_to_lua(b, *db.Get(s)); }
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

void DataBackendLua::pimpl_s::add_data_to_lua(toolbox::LuaObject& lobj, std::string const& key, DataEntity const& v) {
    ASSERT(lobj.is_table() || lobj.is_global());

    if (lobj.has(key)) {
        set_data_to_lua(lobj, key, v);
        return;
    }
    if (v.isTable()) {
        auto const& db = dynamic_cast<DataTable const&>(v);
        auto b = lobj.new_table(key, 0, db.size());
        db.Accept([&](std::string const& k, std::shared_ptr<DataEntity> p) { set_data_to_lua(b, k, *p); });
    } else if (v.isArray()) {
        auto const& db = dynamic_cast<DataArray const&>(v);
        auto b = lobj.new_table(key, db.size(), 0);
        for (int s = 0, se = static_cast<int>(v.size()); s < se; ++s) { add_data_to_lua(b, *db.Get(s)); }
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
    DataBackendLua::pimpl_s::set_data_to_lua(m_pimpl_->m_lua_obj_, key, *v);
    return true;
}
bool DataBackendLua::Set(id_type key, std::shared_ptr<DataEntity> const&) {}

bool DataBackendLua::Add(std::string const& key, std::shared_ptr<DataEntity> const& v) { return false; }
bool DataBackendLua::Add(id_type key, std::shared_ptr<DataEntity> const&) {}

size_type DataBackendLua::Delete(std::string const& key) { return 0; }
size_type DataBackendLua::Delete(id_type key) {}
void DataBackendLua::DeleteAll() {}
size_type DataBackendLua::size() const { return 0; }
size_type DataBackendLua::Accept(std::function<void(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
    if (m_pimpl_->m_lua_obj_.is_global()) {
        UNSUPPORTED;
        UNIMPLEMENTED;
    } else {
        for (auto const& item : m_pimpl_->m_lua_obj_) {
            f(item.first.as<std::string>(), DataBackendLua::pimpl_s::make_data_entity_lua(item.second));
        };
    }
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