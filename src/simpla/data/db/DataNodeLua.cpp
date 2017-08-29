//
// Created by salmon on 17-3-2.
//
#include "DataNodeLua.h"
#include <simpla/data/DataFunction.h>

#include "../DataEntity.h"
#include "../DataNode.h"
#include "../DataTraits.h"

#include "../DataUtilities.h"
#include "LuaObject.h"
namespace simpla {
namespace data {
REGISTER_CREATOR(DataNodeLua, lua);
struct DataFunctionLua : public DataFunction {
    SP_DEFINE_FANCY_TYPE_NAME(DataFunctionLua, DataFunction);
    std::shared_ptr<LuaObject> m_lua_obj_;

   protected:
    DataFunctionLua() = default;
    explicit DataFunctionLua(std::shared_ptr<LuaObject> const& lobj) : m_lua_obj_(lobj) {}

   public:
    ~DataFunctionLua() override = default;
    DataFunctionLua(DataFunctionLua const& other) = delete;
    DataFunctionLua(DataFunctionLua&& other) = delete;

    template <typename... Args>
    static std::shared_ptr<DataFunctionLua> New(Args&&... args) {
        return std::shared_ptr<DataFunctionLua>(new DataFunctionLua(std::forward<Args>(args)...));
    }
    std::shared_ptr<DataEntity> eval(std::initializer_list<std::shared_ptr<DataEntity>> const& args) const override {
        return DataEntity::New();
    };
};
struct DataNodeLua::pimpl_s {
    std::shared_ptr<LuaObject> m_lua_obj_ = nullptr;
    std::shared_ptr<DataNodeLua> m_parent_ = nullptr;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
    std::string m_key_;
    eNodeType m_node_type_ = DataNode::DN_TABLE;
    std::shared_ptr<LuaObject> LazyLoadingTable(int flag = NEW_IF_NOT_EXIST);
    //    std::shared_ptr<DataEntity> LazyLoadingEntity();
    //    void LazyLoading(int flag = NEW_IF_NOT_EXIST);
};
std::shared_ptr<LuaObject> DataNodeLua::pimpl_s::LazyLoadingTable(int flag) {
    std::shared_ptr<LuaObject> res = m_lua_obj_;
    if (m_entity_ != nullptr) {
    } else if (m_lua_obj_ == nullptr && (flag & NEW_IF_NOT_EXIST) != 0) {
        if (m_parent_ == nullptr) {
            auto tmp = LuaObject::New();
            m_lua_obj_ = tmp->new_table(m_key_);
        } else {
            m_lua_obj_ = m_parent_->m_pimpl_->LazyLoadingTable(flag)->new_table(m_key_);
        }
        res = m_lua_obj_;
        if (m_lua_obj_ == nullptr) { RUNTIME_ERROR << "Empty table"; }
    }
    return res;
}
enum { T_NULL = 0, T_INTEGRAL = 0b00001, T_FLOATING = 0b00010, T_STRING = 0b00100, T_BOOLEAN = 0b01000, T_NAN = -1 };

DataNodeLua::DataNodeLua() : m_pimpl_(new pimpl_s) {}
DataNodeLua::DataNodeLua(pimpl_s* pimpl) : m_pimpl_(pimpl) {}
DataNodeLua::~DataNodeLua() {
    m_pimpl_->m_parent_.reset();
    m_pimpl_->m_lua_obj_.reset();
    m_pimpl_->m_parent_.reset();
    delete m_pimpl_;
}
int DataNodeLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    m_pimpl_->m_key_ = query.empty() ? "_ROOT_" : query;
    if (!path.empty()) { m_pimpl_->LazyLoadingTable()->parse_file(path); }
    return SP_SUCCESS;
}
int DataNodeLua::Disconnect() { return 0; }

int DataNodeLua::Parse(std::string const& str) {
    ASSERT(m_pimpl_->m_entity_ == nullptr);
    m_pimpl_->LazyLoadingTable()->parse_string(str);
    return SP_SUCCESS;
};

bool DataNodeLua::isValid() const {
    return !(m_pimpl_->m_lua_obj_ != nullptr && m_pimpl_->m_entity_ == nullptr && m_pimpl_->m_key_.empty());
}

std::shared_ptr<DataNode> DataNodeLua::Duplicate() const {
    auto res = DataNodeLua::New();
    res->m_pimpl_->m_lua_obj_ = m_pimpl_->m_lua_obj_;
    res->m_pimpl_->m_key_ = m_pimpl_->m_key_;
    res->m_pimpl_->m_parent_ = m_pimpl_->m_parent_;
    res->m_pimpl_->m_entity_ = m_pimpl_->m_entity_;

    return res;
}
size_type DataNodeLua::size() const {
    return m_pimpl_->m_lua_obj_ != nullptr ? m_pimpl_->m_lua_obj_->size() : (m_pimpl_->m_entity_ != nullptr ? 1 : 0);
}

DataNode::eNodeType DataNodeLua::type() const {
    eNodeType res = DN_NULL;
    if (m_pimpl_->m_lua_obj_ == nullptr) {
        res = m_pimpl_->m_entity_ == nullptr ? DN_NULL : DN_ENTITY;
    } else if (m_pimpl_->m_lua_obj_->is_array()) {
        res = DN_ARRAY;
    } else if (m_pimpl_->m_lua_obj_->is_table()) {
        res = DN_TABLE;
    } else if (m_pimpl_->m_lua_obj_->is_function()) {
        res = DN_FUNCTION;
    } else if (m_pimpl_->m_lua_obj_->is_lightuserdata()) {
        FIXME << "Unknown Lua object: light user data";
    } else {
        FIXME << "Unknown Lua object type!";
    }
    return res;
}

std::shared_ptr<DataNode> DataNodeLua::Root() const {
    return m_pimpl_->m_parent_ != nullptr ? m_pimpl_->m_parent_->Root() : Self();
}
std::shared_ptr<DataNode> DataNodeLua::Parent() const { return m_pimpl_->m_parent_; }

std::shared_ptr<DataNodeLua> LUAGetNode(std::shared_ptr<LuaObject> const& lobj) {
    std::shared_ptr<DataNodeLua> res = DataNodeLua::New();
    if (lobj == nullptr) {
    } else if (lobj->is_function()) {
        res->m_pimpl_->m_entity_ = DataFunctionLua::New(lobj);
    } else if (lobj->is_table()) {
        res->m_pimpl_->m_lua_obj_ = lobj;
    } else if (lobj->is_boolean()) {
        res->m_pimpl_->m_entity_ = DataLightT<bool>::New(lobj->as<bool>());
    } else if (lobj->is_floating_point()) {
        res->m_pimpl_->m_entity_ = DataLightT<double>::New(lobj->as<double>());
    } else if (lobj->is_integer()) {
        res->m_pimpl_->m_entity_ = DataLightT<int>::New(lobj->as<int>());
    } else if (lobj->is_string()) {
        res->m_pimpl_->m_entity_ = DataLightT<std::string>::New(lobj->as<std::string>());
    } else if (lobj->is_array()) {
        size_type rank;
        auto* extents = new size_type[MAX_NDIMS_OF_ARRAY];
        auto type = lobj->get_shape(&rank, extents);
        if (type == typeid(bool).hash_code()) {
            auto tmp = DataLightT<bool*>::New(rank, extents);
            lobj->get_value(tmp->pointer(), &rank, extents);
            res->m_pimpl_->m_entity_ = tmp;
        } else if (type == typeid(int).hash_code()) {
            auto tmp = DataLightT<int*>::New(rank, extents);
            lobj->get_value(tmp->pointer(), &rank, extents);
            res->m_pimpl_->m_entity_ = tmp;
        } else if (type == typeid(double).hash_code()) {
            auto tmp = DataLightT<double*>::New(rank, extents);
            lobj->get_value(tmp->pointer(), &rank, extents);
            res->m_pimpl_->m_entity_ = tmp;
        } else if (type == typeid(std::string).hash_code()) {
            auto tmp = DataLightT<std::string*>::New(rank, extents);
            lobj->get_value(tmp->pointer(), &rank, extents);
            res->m_pimpl_->m_entity_ = tmp;
        } else {
            res->m_pimpl_->m_lua_obj_ = lobj;
        }
        delete[] extents;
    } else {
        res->m_pimpl_->m_entity_ = nullptr;
        //        WARNING << "Can not convert Lua data :" << key << "[" << lobj->get_typename() << "]" << std::endl;
    }

    return res;
}

size_type DataNodeLua::Foreach(
    std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& fun) const {
    size_type count = 0;
    if (m_pimpl_->m_lua_obj_ != nullptr) {
        for (auto p : *m_pimpl_->m_lua_obj_) { count += fun(p.first->as<std::string>(), LUAGetNode(p.second)); }
    }
    return count;
}

std::shared_ptr<const DataNode> DataNodeLua::Get(std::string const& uri) const {
    if (uri.empty()) { return nullptr; }

    std::shared_ptr<const DataNode> res = nullptr;

    auto pos = uri.find(SP_URL_SPLIT_CHAR);
    if (pos == 0) {
        res = Root()->Get(uri.substr(1));
    } else if (pos != std::string::npos) {
        auto tmp = Get(uri.substr(0, pos));
        res = tmp == nullptr ? nullptr : tmp->Get(uri.substr(pos + 1));
    } else if (m_pimpl_->m_lua_obj_ == nullptr) {
        FIXME << " Get [" << uri << "] from empty table! ";
        res = nullptr;
    } else {
        auto tmp = LUAGetNode(m_pimpl_->m_lua_obj_->get(uri));
        tmp->m_pimpl_->m_parent_ = Self();
        tmp->m_pimpl_->m_key_ = uri;
    };

    return res;
}

size_type DataNodeLua::Set(std::string const& uri, std::shared_ptr<DataEntity> const& entity) {
    if (uri.empty() || entity == nullptr) { return 0; }

    size_type count = 0;
    auto pos = uri.find(SP_URL_SPLIT_CHAR);
    if (pos == 0) {
        count = Root()->Set(uri.substr(1), entity);
    } else if (pos != std::string::npos) {
        auto tmp = DataNodeLua::New();
        tmp->m_pimpl_->m_parent_ = Self();
        tmp->m_pimpl_->m_key_ = uri.substr(0, pos);
        m_pimpl_->m_node_type_ = DN_TABLE;
        count = tmp->Set(uri.substr(pos + 1), entity);
    } else {
        auto lobj = m_pimpl_->LazyLoadingTable();

#define DEFINE_MULTIMETHOD(_T_)                                              \
    else if (auto p = std::dynamic_pointer_cast<DataLightT<_T_>>(entity)) {  \
        count = lobj->set(uri, p->pointer(), 0, nullptr);       \
    }                                                                        \
    else if (auto p = std::dynamic_pointer_cast<DataLightT<_T_*>>(entity)) { \
        size_type rank = p->rank();                                          \
        size_type extents[MAX_NDIMS_OF_ARRAY];                               \
        p->extents(extents);                                                 \
        count = lobj->set(uri, p->pointer(), rank, extents);    \
    }

        if (auto p = std::dynamic_pointer_cast<DataBlock>(entity)) {
            count = 0;
            FIXME << "LUA backend do not support DataBlock ";
        }
        DEFINE_MULTIMETHOD(std::string)
        DEFINE_MULTIMETHOD(bool)
        DEFINE_MULTIMETHOD(float)
        DEFINE_MULTIMETHOD(double)
        DEFINE_MULTIMETHOD(int)
        DEFINE_MULTIMETHOD(long)
        DEFINE_MULTIMETHOD(unsigned int)
        DEFINE_MULTIMETHOD(unsigned long)

        else {
            count = 0;
        }
#undef MULTIMETHOD_T_
    }

    return count;
}
size_type DataNodeLua::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) { return 0; }

size_type DataNodeLua::Delete(std::string const& uri) { return 0; /*m_pimpl_->m_lua_obj_->erase(uri);*/ }

size_type DataNodeLua::Set(index_type s, std::shared_ptr<DataEntity> const& v) { return 0; }

size_type DataNodeLua::Add(index_type s, std::shared_ptr<DataEntity> const& v) { return 0; }

size_type DataNodeLua::Delete(index_type s) { return 0; /*m_pimpl_->m_lua_obj_->erase(uri);*/ }

std::shared_ptr<const DataNode> DataNodeLua::Get(index_type s) const {
    std::shared_ptr<const DataNode> res = nullptr;

    //    auto tmp = LUAGetNode(m_pimpl_->GetLuaObject(flag)->get(s));
    //    tmp->m_pimpl_->m_key_ = std::to_string(s);
    //    tmp->m_pimpl_->m_parent_ = Self();
    //    res = tmp;
    return res;
}
//
std::shared_ptr<DataEntity> DataNodeLua::GetEntity() const { return m_pimpl_->m_entity_; }

//
// int DataNodeLua::SetEntity(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    if (v == nullptr) { m_pimpl_->m_lua_obj_->parse_string(uri); }
//    return 1;
//}
//
// int DataNodeLua::AddEntity(std::string const& key, const std::shared_ptr<DataEntity>& v) {
//    UNIMPLEMENTED;
//    return 0;
//}
// int DataNodeLua::Delete(std::string const& key) {
//    UNIMPLEMENTED;
//    return 0;
//}
// bool DataNodeLua::isNull() const { return m_pimpl_->m_lua_obj_->is_nil(); }
//
// int DataNodeLua::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
//    int counter = 0;
//    if (m_pimpl_->m_lua_obj_->is_global()) {
//        UNSUPPORTED;
//    } else {
//        for (auto const& item : m_pimpl_->m_lua_obj_) {
//            if (item.first.is_string()) {
//                counter += f(item.first.as<std::string>(), this->GetEntity(item.first.as<std::string>()));
//            }
//        };
//    }
//    return counter;
//}
//
// std::shared_ptr<DataEntity> DataNodeLua::Serialize(std::string const& url) {
//    auto obj = m_pack_->m_lua_obj_->get(url);
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto database = std::make_shared<DataNodeLua>();
//        database->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(database));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << ":" << obj.get_typename() << std::endl;
//    }
//};
// std::shared_ptr<DataEntity> DataNodeLua::Serialize(std::string const& url) const {
//    auto obj = m_pack_->m_lua_obj_->get(url);
//    ASSERT(!obj.empty());
//    if (obj.is_floating_point()) {
//        return std::make_shared<DataEntityLua<double>>(obj);
//    } else if (obj.is_integer()) {
//        return std::make_shared<DataEntityLua<int>>(obj);
//    } else if (obj.is_string()) {
//        return std::make_shared<DataEntityLua<std::string>>(obj);
//    } else if (obj.is_table()) {
//        auto database = std::make_shared<DataNodeLua>();
//        database->m_pack_->m_lua_obj_ = obj;
//        return std::dynamic_pointer_cast<DataEntity>(std::make_shared<DataTable>(database));
//    } else {
//        RUNTIME_ERROR << "Parse error! url=" << url << std::endl;
//    }
//}

}  // namespace data{
}  // namespace simpla