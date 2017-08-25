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
    LuaObject m_lua_obj_;

   protected:
    DataFunctionLua() = default;
    explicit DataFunctionLua(LuaObject const& lobj) : m_lua_obj_(lobj) {}

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
    std::shared_ptr<DataNodeLua> m_parent_ = nullptr;
    LuaObject m_lua_obj_;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
    std::string m_key_{"_ROOT_"};
    pimpl_s() {}
    ~pimpl_s() {}
    pimpl_s(LuaObject l, std::shared_ptr<DataNodeLua> p) : m_lua_obj_(std::move(l)), m_parent_(std::move(p)) {}
};

// template <typename U>
// std::shared_ptr<DataLightT<std::vector<U>>> make_data_ntuple(LuaObject const& lobj) {
//    auto res = DataLightT<std::vector<U>>::New();
//    for (int i = 0; i < lobj.size(); ++i) { res->value()[i] = lobj[i].as<U>(); }
//    return res;
//}

enum { T_NULL = 0, T_INTEGRAL = 0b00001, T_FLOATING = 0b00010, T_STRING = 0b00100, T_BOOLEAN = 0b01000, T_NAN = -1 };

DataNodeLua::DataNodeLua() : m_pimpl_(new pimpl_s) {}
DataNodeLua::DataNodeLua(pimpl_s* pimpl) : m_pimpl_(pimpl) {}

DataNodeLua::~DataNodeLua() { delete m_pimpl_; }
int DataNodeLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    m_pimpl_->m_parent_ = DataNodeLua::New();
    m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_.init();

    if (!path.empty()) {
        m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_.parse_file(path);
        m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_["_ROOT_"];
    } else {
        m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_.new_table("_ROOT_");
    }
    m_pimpl_->m_key_ = "_ROOT_";

    return SP_SUCCESS;
}
int DataNodeLua::Disconnect() { return 0; }

int DataNodeLua::Parse(std::string const& str) {
    if (m_pimpl_->m_parent_ == nullptr) {
        m_pimpl_->m_parent_ = DataNodeLua::New();
        m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_.init();
    }
    m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_.parse_string(str);
    m_pimpl_->m_lua_obj_ = m_pimpl_->m_parent_->m_pimpl_->m_lua_obj_["_ROOT_"];
    return SP_SUCCESS;
};
std::istream& DataNodeLua::Parse(std::istream& is) {
    Parse(std::string(std::istreambuf_iterator<char>(is), {}));
    return is;
}
bool DataNodeLua::isValid() const {
    return !(!m_pimpl_->m_lua_obj_.is_null() && m_pimpl_->m_entity_ == nullptr && m_pimpl_->m_key_.empty());
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
size_type DataNodeLua::GetNumberOfChildren() const { return m_pimpl_->m_lua_obj_.size(); }

DataNode::eNodeType DataNodeLua::NodeType() const {
    eNodeType res = DN_NULL;
    if (m_pimpl_->m_lua_obj_.is_null()) {
        res = m_pimpl_->m_entity_ == nullptr ? DN_NULL : DN_ENTITY;
    } else if (m_pimpl_->m_lua_obj_.is_array()) {
        res = DN_ARRAY;
    } else if (m_pimpl_->m_lua_obj_.is_table()) {
        res = DN_TABLE;
    } else if (m_pimpl_->m_lua_obj_.is_function()) {
        res = DN_FUNCTION;
    } else if (m_pimpl_->m_lua_obj_.is_boolean() || m_pimpl_->m_lua_obj_.is_string() ||
               m_pimpl_->m_lua_obj_.is_number()) {
        res = DN_ENTITY;
    } else if (m_pimpl_->m_lua_obj_.is_lightuserdata()) {
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

int get_array_shape(LuaObject const& lobj, int level, std::vector<size_type>& extents) {
    ASSERT(level < MAX_NDIMS_OF_ARRAY);
    int type = T_NULL;
    if (lobj.is_array()) {
        while (extents.size() <= level) { extents.push_back(0); }
        extents[level] = std::max(extents[level], lobj.size());
        for (int i = 0; i < lobj.size(); ++i) { type = type | get_array_shape(lobj[i + 1], level + 1, extents); }
    } else if (lobj.is_string()) {
        type = type | T_STRING;
    } else if (lobj.is_integer()) {
        type = type | T_INTEGRAL;
    } else if (lobj.is_floating_point()) {
        type = type | T_FLOATING;
    } else if (lobj.is_boolean()) {
        type = type | T_BOOLEAN;
    } else {
        type = type | (-1);
    }
    return type;
}
template <typename U>
U* set_array_snan(U* data, int level, std::vector<size_type> const& extents) {
    static U snan = std::numeric_limits<U>::signaling_NaN();
    if (level == extents.size() - 1) {
        for (size_type i = 0, ie = extents[level]; i < ie; ++i) { data[i] = snan; }
        data += extents[level];
    } else if (level < extents.size() - 1) {
        for (size_type i = 0, ie = extents[i]; i < ie; ++i) { data = set_array_snan(data, level + 1, extents); }
    }
    return data;
}
template <typename U>
U* get_array_lua(LuaObject const& lobj, U* data, int level, std::vector<size_type> const& extents) {
    static U snan = std::numeric_limits<U>::signaling_NaN();
    if (level == extents.size() - 1) {
        for (size_type i = 0, ie = std::min(lobj.size(), extents[level]); i < ie; ++i) { data[i] = lobj[i].as<U>(); }
        for (size_type i = lobj.size(), ie = extents[level]; i < ie; ++i) { data[i] = snan; }
        data += extents[level];
    } else if (level < extents.size() - 1) {
        for (size_type i = 0, ie = std::min(lobj.size(), extents[i]); i < ie; ++i) {
            data = get_array_lua(lobj[i], data, level + 1, extents);
        }
    }
    return data;
}

std::shared_ptr<DataNodeLua> MakeDataNodeLua(LuaObject const& lobj) {
    std::shared_ptr<DataNodeLua> res = DataNodeLua::New();

    if (lobj.is_table()) {
        res->m_pimpl_->m_lua_obj_ = lobj;
    } else if (lobj.is_array()) {
        //        std::vector<size_type> extents;
        //        int type = get_array_shape(lobj, 0, extents);
        //        CHECK(key);
        //        CHECK(lobj.size());
        //        CHECK(extents);
        //        switch (type) {
        //            case T_INTEGRAL: {
        //                auto p = DataLightT<int*>::New(extents.size(), &extents[0]);
        //                get_array_lua(lobj, p->value().get(), 0, extents);
        //                res->m_pimpl_->m_entity_ = p;
        //            } break;
        //            case T_FLOATING: {
        //                auto p = DataLightT<double*>::New(extents.size(), &extents[0]);
        //                get_array_lua(lobj, p->value().get(), 0, extents);
        //                res->m_pimpl_->m_entity_ = p;
        //            } break;
        //            case T_BOOLEAN: {
        //                auto p = DataLightT<bool*>::New(extents.size(), &extents[0]);
        //                get_array_lua(lobj, p->value().get(), 0, extents);
        //                res->m_pimpl_->m_entity_ = p;
        //            } break;
        //            case T_STRING: {
        //                //            auto p = DataLightT<std::string*>::New(ndims, extents);
        //                //            get_array_lua(lobj, &p->value()[0], ndims, extents);
        //                //            res = p;
        //                FIXME;
        //
        //            } break;
        //            default:
        //                break;
        //        }

    } else if (lobj.is_boolean()) {
        res->m_pimpl_->m_entity_ = DataLightT<bool>::New(lobj.as<bool>());
    } else if (lobj.is_floating_point()) {
        res->m_pimpl_->m_entity_ = DataLightT<double>::New(lobj.as<double>());
    } else if (lobj.is_integer()) {
        res->m_pimpl_->m_entity_ = DataLightT<int>::New(lobj.as<int>());
    } else if (lobj.is_string()) {
        res->m_pimpl_->m_entity_ = DataLightT<std::string>::New(lobj.as<std::string>());
    } else if (lobj.is_function()) {
        res->m_pimpl_->m_entity_ = DataFunctionLua::New(lobj);
    } else {
        res->m_pimpl_->m_entity_ = DataEntity::New();
        //        WARNING << "Can not convert Lua data :" << key << "[" << lobj.get_typename() << "]" << std::endl;
    }

    return res;
}

int DataNodeLua::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) {
    int count = 0;
    for (auto p : m_pimpl_->m_lua_obj_) { count += fun(p.first.as<std::string>(), MakeDataNodeLua(p.second)); }
    return count;
}
int DataNodeLua::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    int count = 0;
    if (!m_pimpl_->m_lua_obj_.is_null()) {
        for (auto p : m_pimpl_->m_lua_obj_) { count += fun(p.first.as<std::string>(), MakeDataNodeLua(p.second)); }
    }
    return count;
}

std::shared_ptr<DataNode> DataNodeLua::GetNode(std::string const& uri, int flag) {
    std::shared_ptr<DataNodeLua> res = nullptr;
    if ((flag & RECURSIVE) != 0) {
        res = std::dynamic_pointer_cast<DataNodeLua>(RecursiveFindNode(shared_from_this(), uri, flag).second);
    } else {
        if (m_pimpl_->m_lua_obj_.is_null() && (flag & NEW_IF_NOT_EXIST) != 0) {
            m_pimpl_->m_lua_obj_ =
                std::dynamic_pointer_cast<DataNodeLua>(Parent())->m_pimpl_->m_lua_obj_.new_table(m_pimpl_->m_key_);
        }
        if (m_pimpl_->m_lua_obj_.is_null()) {
            res = DataNodeLua::New();
            FIXME << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = MakeDataNodeLua(m_pimpl_->m_lua_obj_[uri]);
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
        if (m_pimpl_->m_lua_obj_.is_null()) {
            FIXME << "Can not get object [" << uri << "] from null group !" << std::endl;
        } else {
            res = MakeDataNodeLua(m_pimpl_->m_lua_obj_[uri]);
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
int DataNodeLua::DeleteNode(std::string const& uri, int flag) { return 0; /*m_pimpl_->m_lua_obj_.erase(uri);*/ }

void DataNodeLua::Clear() {}

std::shared_ptr<DataEntity> DataNodeLua::Get() const { return m_pimpl_->m_entity_; }

int DataNodeLua::Set(std::shared_ptr<DataEntity> const& entity) {
    int count = 0;

    if (entity == nullptr) {
        FIXME << " Set value : " << m_pimpl_->m_key_ << " = " << *entity << std::endl;
    } else if (auto parent = std::dynamic_pointer_cast<DataNodeLua>(Parent())) {
        auto& lobj = parent->m_pimpl_->m_lua_obj_;
        if (auto p = std::dynamic_pointer_cast<DataLightT<std::string>>(entity)) {
            count = lobj.set(m_pimpl_->m_key_, p->value());
        } else if (auto p = std::dynamic_pointer_cast<DataLightT<std::string*>>(entity)) {
        } else if (auto p = std::dynamic_pointer_cast<DataLight>(entity)) {
            if (false) {}
#define DEC_TYPE(_T_)                                     \
    else if (entity->value_type_info() == typeid(_T_)) {  \
        count = lobj.set(m_pimpl_->m_key_, p->as<_T_>()); \
    }
            DEC_TYPE(bool)
            DEC_TYPE(float)
            DEC_TYPE(double)
            DEC_TYPE(int)
            DEC_TYPE(long)
            DEC_TYPE(unsigned int)
            DEC_TYPE(unsigned long)
#undef DEC_TYPE

        } else if (auto p = std::dynamic_pointer_cast<DataBlock>(entity)) {
            FIXME << "LUA backend do not support DataBlock ";
        }
    }
    return count;
}
int DataNodeLua::Add(std::shared_ptr<DataEntity> const& v) { return AddNode()->Set(v); }

//

//
// int DataNodeLua::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    if (v == nullptr) { m_pimpl_->m_lua_obj_.parse_string(uri); }
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
// bool DataNodeLua::isNull() const { return m_pimpl_->m_lua_obj_.is_nil(); }
//
// int DataNodeLua::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
//    int counter = 0;
//    if (m_pimpl_->m_lua_obj_.is_global()) {
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
//    auto obj = m_pack_->m_lua_obj_.get(url);
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
//    auto obj = m_pack_->m_lua_obj_.get(url);
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