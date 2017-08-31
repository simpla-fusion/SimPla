//
// Created by salmon on 17-3-6.
//
#include "DataNodeMemory.h"
#include <iomanip>
#include <map>
#include <regex>
#include "../DataBlock.h"
#include "../DataEntity.h"
#include "../DataNode.h"
#include "DataNodeExt.h"
namespace simpla {
namespace data {

struct DataNodeEntityMemory : public DataNodeEntity {
    SP_DATA_NODE_ENTITY_HEAD(DataNodeEntityMemory)
   protected:
    explicit DataNodeEntityMemory(std::shared_ptr<DataEntity> const& v);

   private:
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
};
// template <typename... Args>
// DataNodeEntityMemory::DataNodeEntityMemory(Args&&... args) : m_entity_(make_data(std::forward<Args>(args)...)) {}
DataNodeEntityMemory::DataNodeEntityMemory() : m_entity_(nullptr) {}
DataNodeEntityMemory::DataNodeEntityMemory(std::shared_ptr<DataEntity> const& v) : m_entity_(v) {}
DataNodeEntityMemory::~DataNodeEntityMemory() = default;

std::shared_ptr<DataEntity> DataNodeEntityMemory::GetEntity() const { return m_entity_; }
size_type DataNodeEntityMemory::SetEntity(std::shared_ptr<DataEntity> const& entity) {
    m_entity_ = entity;
    return 1;
};
struct DataNodeFunctionMemory : public DataNodeFunction {
    SP_DATA_NODE_FUNCTION_HEAD(DataNodeFunctionMemory);
};
DataNodeFunctionMemory::DataNodeFunctionMemory() {}
DataNodeFunctionMemory::~DataNodeFunctionMemory() {}

struct DataNodeArrayMemory : public DataNodeArray {
    SP_DATA_NODE_ARRAY_HEAD(DataNodeArrayMemory)
   private:
    std::vector<std::shared_ptr<DataNode>> m_data_;
};

DataNodeArrayMemory::DataNodeArrayMemory() = default;
DataNodeArrayMemory::~DataNodeArrayMemory() = default;

size_type DataNodeArrayMemory::size() const { return m_data_.size(); }
std::shared_ptr<DataNode> DataNodeArrayMemory::CreateChild() const {
    auto node = DataNodeMemory::New();
    node->m_parent_ = Self();
    return node;
}
size_type DataNodeArrayMemory::Set(size_type s, std::shared_ptr<DataNode> const& v) {
    if (s >= size()) { m_data_.resize(s + 1); }
    m_data_[s] = (v);
    return 1;
}
size_type DataNodeArrayMemory::Add(size_type s, std::shared_ptr<DataNode> const& v) {
    if (s >= size()) { return 0; }
    if (auto p = std::dynamic_pointer_cast<DataNodeArrayMemory>(m_data_[s])) { return p->Set(p->size(), v); }
    return 0;
}
size_type DataNodeArrayMemory::Delete(size_type s) {
    FIXME;
    return 0;
}

size_type DataNodeArrayMemory::Add(std::shared_ptr<DataNode> const& v) {
    m_data_.push_back(v);
    return 1;
};

std::shared_ptr<DataNode> DataNodeArrayMemory::Get(size_type s) const { return m_data_.at(s); }

struct DataNodeTableMemory : public DataNodeTable {
    SP_DATA_NODE_TABLE_HEAD(DataNodeTableMemory)
    std::shared_ptr<DataNode> CreateTable() const override;
    std::shared_ptr<DataNode> CreateArray() const override;
    std::shared_ptr<DataNode> CreateEntity(std::shared_ptr<DataEntity> const& v) const override;
    std::shared_ptr<DataEntity> GetEntity() const override;

   private:
    std::map<std::string, std::shared_ptr<DataNode>> m_table_;
};
DataNodeTableMemory::DataNodeTableMemory() {}
DataNodeTableMemory::~DataNodeTableMemory() {}

std::shared_ptr<DataNode> DataNodeTableMemory::CreateChild() const {
    auto res = DataNodeTableMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeTableMemory::CreateTable() const {
    auto res = DataNodeTableMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeTableMemory::CreateArray() const {
    auto res = DataNodeArrayMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeTableMemory::CreateEntity(std::shared_ptr<DataEntity> const& v) const {
    auto res = DataNodeEntityMemory::New(v);
    res->m_parent_ = Self();
    return res;
};
std::shared_ptr<DataEntity> DataNodeTableMemory::GetEntity() const {
    std::shared_ptr<DataEntity> res = nullptr;
    if (auto p = dynamic_cast<DataNodeEntityMemory const*>(this)) {
        res = p->GetEntity();
    } else {
        res = DataEntity::New();
    }
    return res;
}
size_type DataNodeTableMemory::size() const { return m_table_.size(); }

size_type DataNodeTableMemory::Set(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    size_type tail = 0;
    auto obj = Self();
    std::string k = uri;
    while (obj != nullptr) {
        tail = k.find(SP_URL_SPLIT_CHAR);

        if (tail == std::string::npos) {
            obj->m_table_[k] = v;
            count = 1;
            break;
        } else {
            obj = std::dynamic_pointer_cast<DataNodeTableMemory>(
                obj->m_table_.emplace(k.substr(0, tail), NewTable()).first->second);
            k = k.substr(tail + 1);
        }
    }

    return count;

    //    size_type count = 0;
    //    if (uri.empty()) { return 0; }
    //    auto pos = uri.find(SP_URL_SPLIT_CHAR);
    //    if (pos == 0) {
    //        count = Root()->Set(uri.substr(1), v);
    //    } else {
    //        auto p = m_table_.emplace(uri.substr(0, pos), New());
    //        if (p.second) { p.first->second->m_parent_ = Self(); }
    //        if (pos != std::string::npos) {
    //            count = p.first->second->Set(uri.substr(pos + 1), v);
    //        } else {
    //            //                p.first->second->m_table_.clear();
    //            //                p.first->second->m_entity_ = v;
    //            //                p.first->second->m_node_type_ = DN_ENTITY;
    //            //                count = 1;
    //        }
    //    }
}
size_type DataNodeTableMemory::Add(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    size_type tail = 0;
    std::shared_ptr<DataNode> obj = Self();
    std::string k = uri;
    while (obj != nullptr) {
        auto p = std::dynamic_pointer_cast<DataNodeTableMemory>(obj);
        if (p == nullptr) { break; }
        tail = k.find(SP_URL_SPLIT_CHAR);

        if (tail == std::string::npos) {
            obj = p->m_table_.emplace(k, NewArray()).first->second;
            if (auto q = std::dynamic_pointer_cast<DataNodeArrayMemory>(obj)) { count = q->Add(v); }
            break;
        } else {
            obj = p->m_table_.emplace(k.substr(0, tail), NewTable()).first->second;
            k = k.substr(tail + 1);
        }
    }
    return count;
}
std::shared_ptr<DataNode> DataNodeTableMemory::Get(std::string const& uri) const {
    if (uri.empty()) { return nullptr; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Get(uri.substr(1)); }

    size_type count = 0;
    std::shared_ptr<DataNode> obj = Self();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {
        auto tail = k.find(SP_URL_SPLIT_CHAR);
        if (auto p = std::dynamic_pointer_cast<DataNodeTableMemory>(obj)) {
            auto it = p->m_table_.find(k.substr(0, tail));
            obj = (it != p->m_table_.end()) ? it->second : nullptr;
        } else {
            obj = obj->Get(k.substr(0, tail));
        }
        if (tail != std::string::npos) {
            k = k.substr(tail + 1);
        } else {
            k = "";
        };
    }
    return obj;
};
size_type DataNodeTableMemory::Delete(std::string const& uri) {
    size_type count = 0;
    if (uri.empty()) {
    } else {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            count = Root()->Delete(uri.substr(1));
        } else {
            auto it = m_table_.find(uri.substr(0, pos));
            if (it != m_table_.end()) {
                if (pos == std::string::npos) {
                    m_table_.erase(it);
                    count = 1;
                } else {
                    count = it->second->Delete(uri.substr(pos));
                };
            }
        }
    }
    return count;
}
size_type DataNodeTableMemory::Foreach(
    std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const {
    size_type count = 0;
    for (auto const& item : m_table_) { count += f(item.first, item.second); }
    return count;
}

DataNodeMemory::DataNodeMemory() {}
DataNodeMemory::~DataNodeMemory() {}
std::shared_ptr<DataNode> DataNodeMemory::New() { return DataNodeTableMemory::New(); }
std::shared_ptr<DataNode> DataNodeMemory::CreateEntity(std::shared_ptr<DataEntity> const& v) const {
    auto res = DataNodeEntityMemory::New(v);
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeMemory::CreateTable() const {
    auto res = DataNodeTableMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeMemory::CreateArray() const {
    auto res = DataNodeArrayMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNode> DataNodeMemory::CreateFunction() const {
    auto res = DataNodeFunctionMemory::New();
    res->m_parent_ = Self();
    return res;
}

REGISTER_CREATOR(DataNodeMemory, mem);

}  // namespace data {
}  // namespace simpla{