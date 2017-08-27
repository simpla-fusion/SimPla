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
    std::string m_key_{"_ROOT_"};
};

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
    std::string root_s = query.empty() ? "_ROOT_" : query;
    m_pimpl_->m_parent_ = DataNodeLua::New();
    m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_ = LuaObject::New();

    if (!path.empty()) {
        m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->parse_file(path);
        m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->get(root_s);
    } else {
        m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->new_table(root_s);
    }
    m_pimpl_->m_key_ = root_s;

    return SP_SUCCESS;
}
int DataNodeLua::Disconnect() { return 0; }

int DataNodeLua::Parse(std::string const& str) {
    if (m_pimpl_->m_parent_ == nullptr) {
        m_pimpl_->m_parent_ = DataNodeLua::New();
        m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->init();
    }
    m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->parse_string(str);
    m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_->get("_ROOT_");
    return SP_SUCCESS;
};
std::istream& DataNodeLua::Parse(std::istream& is) {
    Parse(std::string(std::istreambuf_iterator<char>(is), {}));
    return is;
}
bool DataNodeLua::isValid() const {
    return !(m_pimpl_->m_lua_obj_ != nullptr && m_pimpl_->m_entity_ == nullptr && m_pimpl_->m_key_.empty());
}
int DataNodeLua::Flush() { return 0; }

std::shared_ptr<DataNode> DataNodeLua::Duplicate() const {
    auto res = DataNodeLua::New();
    res->m_pimpl_->m_lua_obj_ = m_pimpl_->m_lua_obj_;
    res->m_pimpl_->m_key_ = m_pimpl_->m_key_;
    res->m_pimpl_->m_parent_ = m_pimpl_->m_parent_;
    res->m_pimpl_->m_entity_ = m_pimpl_->m_entity_;

    return res;
}
size_type DataNodeLua::GetNumberOfChildren() const {
    return m_pimpl_->m_lua_obj_ == nullptr ? 0 : m_pimpl_->m_lua_obj_->size();
}

DataNode::eNodeType DataNodeLua::NodeType() const {
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
int DataNodeLua::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) {
    int count = 0;
    for (auto p : *m_pimpl_->m_lua_obj_) { count += fun(p.first->as<std::string>(), LUAGetNode(p.second)); }
    return count;
}
int DataNodeLua::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    int count = 0;
    if (m_pimpl_->m_lua_obj_ != nullptr) {
        for (auto p : *m_pimpl_->m_lua_obj_) { count += fun(p.first->as<std::string>(), LUAGetNode(p.second)); }
    }
    return count;
}

std::shared_ptr<DataNode> DataNodeLua::GetNode(std::string const& uri, int flag) {
    std::shared_ptr<DataNodeLua> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeLua>(RecursiveFindNode(shared_from_this(), uri, flag).second);
    } else {
        if (m_pimpl_->m_lua_obj_ == nullptr && (flag & NEW_IF_NOT_EXIST) != 0) {
            m_pimpl_->m_lua_obj_ =
                std::dynamic_pointer_cast<DataNodeLua>(Parent())->m_pimpl_->m_lua_obj_->new_table(m_pimpl_->m_key_);
        }
        if (m_pimpl_->m_lua_obj_ == nullptr) {
            res = DataNodeLua::New();
            RUNTIME_ERROR << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = LUAGetNode(m_pimpl_->m_lua_obj_->get(uri));
            res->m_pimpl_->m_key_ = uri;
            res->m_pimpl_->m_parent_ = Self();
        }
    }
    return res;
}
std::shared_ptr<DataNode> DataNodeLua::GetNode(std::string const& uri, int flag) const {
    std::shared_ptr<DataNodeLua> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeLua>(RecursiveFindNode(Self(), uri, flag & (~NEW_IF_NOT_EXIST)).second);
    } else {
        if (m_pimpl_->m_lua_obj_ == nullptr) {
            FIXME << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = LUAGetNode(m_pimpl_->m_lua_obj_->get(uri));
            res->m_pimpl_->m_parent_ = Self();
            res->m_pimpl_->m_key_ = uri;
        }
    }

    return res;
}
std::shared_ptr<DataNode> DataNodeLua::GetNode(index_type s, int flag) { return GetNode(std::to_string(s), flag); }
std::shared_ptr<DataNode> DataNodeLua::GetNode(index_type s, int flag) const {
    return GetNode(std::to_string(s), flag);
}
int DataNodeLua::DeleteNode(std::string const& uri, int flag) { return 0; /*m_pimpl_->m_lua_obj_->erase(uri);*/ }

void DataNodeLua::Clear() {}

std::shared_ptr<DataEntity> DataNodeLua::Get() const { return m_pimpl_->m_entity_; }

int DataNodeLua::Set(std::shared_ptr<DataEntity> const& entity) {
    int count = 0;

    if (entity == nullptr) {
        FIXME << " Set value : " << m_pimpl_->m_key_ << " = " << *entity << std::endl;
    } else if (auto parent = std::dynamic_pointer_cast<DataNodeLua>(Parent())) {
        auto lobj = parent->m_pimpl_->m_lua_obj_;
        if (auto p = std::dynamic_pointer_cast<DataBlock>(entity)) {
            count = 0;
            FIXME << "LUA backend do not support DataBlock ";
        }
#define DEFINE_MULTIMETHOD(_T_)                                              \
    else if (auto p = std::dynamic_pointer_cast<DataLightT<_T_>>(entity)) {  \
        count = lobj->set(m_pimpl_->m_key_, p->pointer(), 0, nullptr);       \
    }                                                                        \
    else if (auto p = std::dynamic_pointer_cast<DataLightT<_T_*>>(entity)) { \
        size_type rank = p->rank();                                          \
        size_type extents[MAX_NDIMS_OF_ARRAY];                               \
        p->extents(extents);                                                 \
        count = lobj->set(m_pimpl_->m_key_, p->pointer(), rank, extents);    \
    }
        DEFINE_MULTIMETHOD(std::string)
        DEFINE_MULTIMETHOD(bool)
        DEFINE_MULTIMETHOD(float)
        DEFINE_MULTIMETHOD(double)
        DEFINE_MULTIMETHOD(int)
        DEFINE_MULTIMETHOD(long)
        DEFINE_MULTIMETHOD(unsigned int)
        DEFINE_MULTIMETHOD(unsigned long)
#undef MULTIMETHOD_T_
        else {
            count = 0;
        }
    }
    return count;
}
int DataNodeLua::Add(std::shared_ptr<DataEntity> const& v) { return AddNode()->Set(v); }

//

//
// int DataNodeLua::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    if (v == nullptr) { m_pimpl_->m_lua_obj_->parse_string(uri); }
//    return 1;
//}
//
// int DataNodeLua::Add(std::string const& key, const std::shared_ptr<DataEntity>& v) {
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
//                counter += f(item.first.as<std::string>(), this->Get(item.first.as<std::string>()));
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