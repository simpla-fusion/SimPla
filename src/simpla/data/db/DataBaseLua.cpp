//
// Created by salmon on 17-3-2.
//
#include "DataBaseLua.h"

#include "../DataEntity.h"
#include "../DataNode.h"
#include "../DataTraits.h"

#include "LuaObject.h"

namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseLua, lua);

struct DataBaseLua::Node : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(Node, DataNode)
    e_NodeType m_node_type = DN_NULL;
    std::shared_ptr<Node> m_parent_ = nullptr;
    std::map<std::string, std::shared_ptr<Node>> m_table_;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;

    LuaObject m_lua_obj_;

   protected:
    Node() = default;
    explicit Node(std::shared_ptr<Node> v) : m_parent_(std::move(v)){};
    explicit Node(std::shared_ptr<DataEntity> v) : m_entity_(std::move(v)){};

    explicit Node(Node const& other) = delete;
    explicit Node(Node&& other) = delete;
    Node& operator=(Node const& other) = delete;
    Node& operator=(Node&& other) = delete;

   public:
    ~Node() override = default;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<Node>(new Node(std::forward<Args>(args)...));
    }

    void Connect(std::string const& authority, std::string const& path, std::string const& query,
                 std::string const& fragment);

    std::shared_ptr<DataNode> Duplicate() const override { return Node::New(m_parent_); }
    size_type GetNumberOfChildren() const override { return m_table_.size(); }

    /** @addtogroup{ Interface */
    int Flush() override { return 0; }
    e_NodeType NodeType() const override { return m_node_type; }

    std::shared_ptr<DataNode> Root() override { return m_parent_ != nullptr ? m_parent_->Root() : shared_from_this(); }
    std::shared_ptr<DataNode> Parent() const override { return m_parent_; }

    int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) override;
    int Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const override;

    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) override;
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const override;
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) override;
    std::shared_ptr<DataNode> GetNode(index_type s, int flag) const override;
    int DeleteNode(std::string const& uri, int flag) override;
    void Clear() override { m_table_.clear(); }

    std::shared_ptr<DataEntity> Get() override { return m_entity_; }
    std::shared_ptr<DataEntity> Get() const override { return m_entity_; }
    int Set(std::shared_ptr<DataEntity> const& v) override;
    int Add(std::shared_ptr<DataEntity> const& v) override;

    LuaObject m_lua_obj_;

    template <typename U>
    std::shared_ptr<DataEntity> make_data_array_lua(LuaObject const& lobj);
    std::shared_ptr<DataEntity> make_data_entity_lua(LuaObject const& lobj);
};

struct DataBaseLua::pimpl_s {
    std::shared_ptr<Node> m_root_ = nullptr;
};
DataBaseLua::DataBaseLua() : m_pimpl_(new pimpl_s) {}
DataBaseLua::~DataBaseLua() { delete m_pimpl_; }
int DataBaseLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    m_pimpl_->m_root_ = Node::New();
    m_pimpl_->m_root_->Connect(authority, path, query, fragment);
    return 0;
}
int DataBaseLua::Disconnect() { return 0; }
bool DataBaseLua::isNull() const { return false; }
int DataBaseLua::Flush() { return 0; }

std::shared_ptr<DataNode> DataBaseLua::Root() { return Node::New(); }

// void DataBaseLua::Parser(std::string const& str) { m_pimpl_->m_lua_obj_.parse_string(str); }

int DataBaseLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    m_pimpl_->m_lua_obj_.parse_file(path);
    return SP_SUCCESS;
}
int DataBaseLua::Disconnect() { return SP_SUCCESS; }
std::shared_ptr<DataNode> DataBaseLua::Root() { return DataNode::New(); }

//
// template <typename U>
// std::shared_ptr<DataEntity> DataBaseLua::pimpl_s::make_data_array_lua(LuaObject const& lobj) {
//    auto res = DataArrayT<U>::New();
//    for (auto const& item : lobj) { res->Add(item.second.as<U>()); }
//    return std::dynamic_pointer_cast<DataEntity>(res);
//}
// std::shared_ptr<DataEntity> DataBaseLua::pimpl_s::make_data_entity_lua(LuaObject const& lobj) {
//    std::shared_ptr<DataEntity> res = nullptr;
//
//    if (lobj.is_table()) {
//        auto p = DataBaseLua::New();
//        p->m_pimpl_->m_lua_obj_ = lobj;
//        res = DataTable::New(p);
//    } else if (lobj.is_array()) {
//        auto a = *lobj.begin();
//        if (a.second.is_integer()) {
//            res = make_data_array_lua<int>(lobj);
//        } else if (a.second.is_floating_point()) {
//            res = make_data_array_lua<double>(lobj);
//        } else if (a.second.is_string()) {
//            res = make_data_array_lua<std::string>(lobj);
//        }
//    } else if (lobj.is_boolean()) {
//        res = make_data(lobj.as<bool>());
//    } else if (lobj.is_floating_point()) {
//        res = make_data<double>(lobj.as<double>());
//    } else if (lobj.is_integer()) {
//        res = make_data<int>(lobj.as<int>());
//    } else if (lobj.is_string()) {
//        res = make_data<std::string>(lobj.as<std::string>());
//    } else {
//        RUNTIME_ERROR << "illegal data type of Lua :" << lobj.get_typename() << std::endl;
//    }
//    return res;
//}
// std::shared_ptr<DataEntity> DataBaseLua::Get(std::string const& key) const {
//    return m_pimpl_->make_data_entity_lua(m_pimpl_->m_lua_obj_.get(key));
//};
//
// int DataBaseLua::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    if (v == nullptr) { m_pimpl_->m_lua_obj_.parse_string(uri); }
//    return 1;
//}
//
// int DataBaseLua::Add(std::string const& key, const std::shared_ptr<DataEntity>& v) {
//    UNIMPLEMENTED;
//    return 0;
//}
// int DataBaseLua::Delete(std::string const& key) {
//    UNIMPLEMENTED;
//    return 0;
//}
// bool DataBaseLua::isNull() const { return m_pimpl_->m_lua_obj_.is_nil(); }
//
// int DataBaseLua::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
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