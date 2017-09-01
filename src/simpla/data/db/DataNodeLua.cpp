//
// Created by salmon on 17-9-1.
//
#include <simpla/data/DataFunction.h>

#include "../DataEntity.h"
#include "../DataNode.h"
#include "../DataTraits.h"
#include "../DataUtilities.h"
#include "DataNodeMemory.h"
#include "LuaObject.h"
namespace simpla {
namespace data {
struct DataNodeLua : public DataNodeMemory {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeLua, DataNodeMemory)
    SP_DATA_NODE_HEAD(DataNodeLua)

   protected:
    explicit DataNodeLua(eNodeType e_type) : DataNodeMemory(e_type) {}

   public:
    int Connect(std::string const& authority, std::string const& path, std::string const& query,
                std::string const& fragment) override;
    int Disconnect() override;
    int Parse(std::string const& str) override;
    int Flush() override;
    size_type Dump(std::string const& path);
    size_type Load(std::shared_ptr<LuaObject> const&);

    std::string const& GetFileName() const { return m_file_name_; }
    void SetFileName(std::string const& s) { m_file_name_ = s; }

   private:
    std::string m_file_name_;
};

REGISTER_CREATOR(DataNodeLua, lua);
DataNodeLua::DataNodeLua() : DataNodeMemory(DataNode::DN_TABLE){};
DataNodeLua::~DataNodeLua() = default;
int DataNodeLua::Connect(std::string const& authority, std::string const& path, std::string const& query,
                         std::string const& fragment) {
    if (!path.empty()) {
        auto tmp = LuaObject::New();
        tmp->parse_file(path);

        Load(tmp->get(query.empty() ? "_ROOT_" : query));
        SetFileName(path);
    }
    return SP_SUCCESS;
}
int DataNodeLua::Disconnect() { return 0; }
int DataNodeLua::Flush() {
    int success = 0;
    if (isRoot()) { success = Dump(GetFileName()); }
    return success;
}
int DataNodeLua::Parse(std::string const& str) {
    auto tmp = LuaObject::New();
    tmp->parse_string(str);
    Load(tmp->get("_ROOT_"));
    return SP_SUCCESS;
};
size_type DataNodeLua::Dump(std::string const& path) {
    TODO << "Dump file to lua file:" << path;
    return 0;
}

std::shared_ptr<DataNode> LuaToDataNode(std::shared_ptr<LuaObject> const& tobj) {
    if (tobj == nullptr) { return DataNode::New(DataEntity::New()); }
    std::shared_ptr<DataNode> res = nullptr;
    switch (tobj->type()) {
        case LuaObject::LUA_T_BOOLEAN:
            res = DataNode::New(DataLight::New(tobj->as<bool>()));
            break;
        case LuaObject::LUA_T_INTEGER:
            res = DataNode::New(DataLight::New(tobj->as<int>()));
            break;
        case LuaObject::LUA_T_FLOATING:
            res = DataNode::New(DataLight::New(tobj->as<double>()));
            break;
        case LuaObject::LUA_T_STRING:
            res = DataNode::New(DataLight::New(tobj->as<std::string>()));
            break;
        case LuaObject::LUA_T_TABLE:
            res = DataNode::New(DataNode::DN_TABLE);
            for (auto const& kv : *tobj) { res->Set(kv.first->as<std::string>(), LuaToDataNode(kv.second)); }
            break;

        case LuaObject::LUA_T_ARRAY: {
            res = DataNode::New(DataNode::DN_ARRAY);
            for (auto const& kv : *tobj) { res->Add(LuaToDataNode(kv.second)); }
            break;
        }
        case LuaObject::LUA_T_FUNCTION:
            res = DataNode::New(DataNode::DN_FUNCTION);
            TODO << " Create Lua Function";
            break;
        default:
            break;
    }
    return res;
}
size_type DataNodeLua::Load(std::shared_ptr<LuaObject> const& lobj) {
    size_type count = 0;
    count = LuaToDataNode(lobj)->Foreach([&](std::string k, std::shared_ptr<DataNode> v) { return this->Set(k, v); });
    return count;
}

}  // namespace data {
}  // namespace simpla {
