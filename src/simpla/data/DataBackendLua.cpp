//
// Created by salmon on 17-3-2.
//
#include "DataBackendLua.h"
#include <simpla/toolbox/LuaObject.h>
#include "DataBackend.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
template <typename U>
struct DataEntityLua : public DataHolder<U> {
    DataEntityLua(DataEntityLua<U> const& other) : DataHolder<U>(other), m_obj_(other.m_obj_){};
    DataEntityLua(toolbox::LuaObject const& v) : m_obj_(v){};
    DataEntityLua(toolbox::LuaObject&& v) : m_obj_(v){};
    virtual ~DataEntityLua(){};

    virtual std::ostream& Print(std::ostream& os, int indent = 0) const {
        os << value();
        return os;
    };
    virtual bool empty() const { return m_obj_.empty(); };
    virtual DataHolderBase* Copy() const { return new DataEntityLua<U>(m_obj_); }
    virtual std::type_info const& type() { return typeid(U); }

    virtual bool equal(U const& v) const { return *DataHolder<U>::pointer() == v; };
    virtual U value() const { return m_obj_.as<U>(); };
    //    virtual U* pointer() const {
    ////        if (m_value_ == nullptr) { m_value_ = new U(m_obj_.as<U>()); }
    //        return m_value_;
    //    };

   private:
    toolbox::LuaObject m_obj_;
};
//
// DataEntityLua::DataEntityLua(toolbox::LuaObject const& u) : m_lua_obj_(u){};
// DataEntityLua::DataEntityLua(toolbox::LuaObject&& u) : m_lua_obj_(u){};
// DataEntityLua::~DataEntityLua() {}
// std::ostream& DataEntityLua::Print(std::ostream& os, int indent) const { return m_lua_obj_.Print(os, indent); };
// bool DataEntityLua::isLight() const { return true; };
//
// std::shared_ptr<DataEntity> DataEntityLua::Copy() const {
//    return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataEntityLua>(m_lua_obj_));
//};
//
// DataBackendLua::DataBackendLua(std::string const& url, std::string const& status) {
//    m_lua_obj_.init();
//    Open(url, status);
//}

struct DataBackendLua::pimpl_s {
    toolbox::LuaObject m_lua_obj_;
};
DataBackendLua::DataBackendLua(std::string const& url, std::string const& status) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_lua_obj_.init();
    if (url != "") Open(url, status);
}
DataBackendLua::DataBackendLua(DataBackendLua const& other) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_lua_obj_ = other.m_pimpl_->m_lua_obj_;
};
DataBackendLua::~DataBackendLua() {
    if (m_pimpl_ != nullptr) { delete m_pimpl_; };
}
void DataBackendLua::Open(std::string const& str, std::string const& status) { m_pimpl_->m_lua_obj_.parse_file(str); }
void DataBackendLua::Parse(std::string const& str) { m_pimpl_->m_lua_obj_.parse_string(str); }
bool DataBackendLua::empty() const { return m_pimpl_->m_lua_obj_.empty(); }
void DataBackendLua::Flush() {}
void DataBackendLua::Close() {}
void DataBackendLua::Clear() {}
void DataBackendLua::Reset() {}
DataBackend* DataBackendLua::Copy() const { return new DataBackendLua(*this); }

std::ostream& DataBackendLua::Print(std::ostream& os, int indent) const {
    return m_pimpl_->m_lua_obj_.Print(os, indent);
}

bool DataBackendLua::Erase(std::string const& key) { UNIMPLEMENTED; }
// std::shared_ptr<DataEntity> DataBackendLua::Set(std::string const& key, std::shared_ptr<DataEntity> const& v) {
//    UNIMPLEMENTED;
//}
std::pair<DataEntity*, bool> DataBackendLua::Insert(std::string const& k){};
std::pair<DataEntity*, bool> DataBackendLua::Insert(std::string const& k, DataEntity const& v, bool assign_is_exists){};
DataEntity* DataBackendLua::Find(std::string const& url) const {}

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