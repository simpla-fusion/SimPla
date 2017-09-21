//
// Created by salmon on 17-9-1.
//
#include <simpla/data/DataFunction.h>
#include <fstream>

#include "../DataEntity.h"
#include "../DataNode.h"
#include "../DataTraits.h"
#include "../DataUtilities.h"
#include "DataNodeMemory.h"
#include "LuaObject.h"
namespace simpla {
namespace data {
struct DataNodeLua : public DataNodeMemory {
    SP_DATA_NODE_HEAD(DataNodeLua, DataNodeMemory)

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
    std::string m_file_name_ = "simpla_unnamed.lua";
};

REGISTER_CREATOR(DataNodeLua, lua);
DataNodeLua::DataNodeLua(DataNode::eNodeType e_type) : base_type(e_type){};
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
    tmp->parse_string("_ROOT_={" + str + "}");
    Load(tmp->get("_ROOT_"));
    return SP_SUCCESS;
};
#define PRINT_ND_ARRAY(_TYPE_)                                                                       \
    else if (auto p = std::dynamic_pointer_cast<const DataLightT<int*>>(entity)) {                   \
        int ndims = p->rank();                                                                       \
        auto extents = new size_type[ndims];                                                         \
        p->extents(extents);                                                                         \
        printNdArray(os, p->pointer(), ndims, extents, true, false, "{", ",", "}", true, 0, indent); \
        delete[] extents;                                                                            \
    }

std::ostream& PrintLua(std::ostream& os, std::shared_ptr<const DataEntity> const& entity, int indent) {
    if (entity->rank() == 0) { entity->Print(os, 0); }
    PRINT_ND_ARRAY(int)
    PRINT_ND_ARRAY(long)
    PRINT_ND_ARRAY(unsigned int)
    PRINT_ND_ARRAY(unsigned long)
    PRINT_ND_ARRAY(float)
    PRINT_ND_ARRAY(double)
    PRINT_ND_ARRAY(bool)
    PRINT_ND_ARRAY(std::string)
    else {
        os << "<N/A>" << std::endl;
    }
    return os;
}

std::ostream& PrintLua(std::ostream& os, std::shared_ptr<DataNode> const& node, int indent) {
    if (node == nullptr) {
        os << "<N/A>";
        return os;
    }
    switch (node->type()) {
        case DataNode::DN_ENTITY: {
            auto entity = node->GetEntity();
            if (entity != nullptr) PrintLua(os, node->GetEntity(), indent + 1);
        } break;
        case DataNode::DN_ARRAY: {
            os << "{ ";
            bool is_first = true;
            bool new_line = node->size() > 1;
            node->Foreach([&](auto k, auto v) {
                if (v == nullptr) { return; }
                if (is_first) {
                    is_first = false;
                } else {
                    os << ", ";
                }
                PrintLua(os, v, indent + 1);
            });

            //            if (new_line) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        } break;
        case DataNode::DN_TABLE: {
            os << "{ ";
            bool is_first = true;
            bool new_line = node->size() > 1;
            node->Foreach([&](auto k, auto v) {
                if (v == nullptr) { return 0; }
                if (is_first) {
                    is_first = false;
                } else {
                    os << ", ";
                }
                if (new_line) { os << std::endl << std::setw(indent + 1) << " "; }
                os << k;
                os << " = ";
                PrintLua(os, v, indent + 1);
                return 1;
            });

            if (new_line) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        } break;
        case DataNode::DN_FUNCTION:
            os << "<FUNCTION>";
            break;
        default:
            os << "<N/A>";
            break;
    }

    return os;
}
size_type DataNodeLua::Dump(std::string const& path) {
    std::ofstream os(path);
    PrintLua(os, shared_from_this(), 0);
    return 1;
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
    LuaToDataNode(lobj)->Foreach([&](std::string k, std::shared_ptr<DataNode> v) { count += this->Set(k, v); });
    return count;
}

}  // namespace data {
}  // namespace simpla {
