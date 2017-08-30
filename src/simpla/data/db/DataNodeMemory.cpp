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
std::shared_ptr<DataNode> DataNodeArrayMemory::NewChild() const {
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
std::shared_ptr<const DataNode> DataNodeArrayMemory::Get(size_type s) const { return m_data_.at(s); }

struct DataNodeTableMemory : public DataNodeTable {
    SP_DATA_NODE_TABLE_HEAD(DataNodeTableMemory)

   private:
    std::map<std::string, std::shared_ptr<DataNode>> m_table_;
};
DataNodeTableMemory::DataNodeTableMemory() {}
DataNodeTableMemory::~DataNodeTableMemory() {}

std::shared_ptr<DataNode> DataNodeTableMemory::NewChild() const {
    auto res = DataNodeTableMemory::New();
    res->m_parent_ = Self();
    return res;
}
size_type DataNodeTableMemory::size() const { return m_table_.size(); }

size_type DataNodeTableMemory::Set(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    size_type count = 0;
    if (!uri.empty()) {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            count = Root()->Set(uri.substr(1), v);
        } else {
            auto p = m_table_.emplace(uri.substr(0, pos), New());
            if (p.second) { p.first->second->m_parent_ = Self(); }
            if (pos != std::string::npos) {
                count = p.first->second->Set(uri.substr(pos + 1), v);
            } else {
                //                p.first->second->m_table_.clear();
                //                p.first->second->m_entity_ = v;
                //                p.first->second->m_node_type_ = DN_ENTITY;
                //                count = 1;
            }
        }
    }
    return count;
}
size_type DataNodeTableMemory::Add(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    //    return base_type::Add(uri, v);
    size_type count = 0;
    if (!uri.empty()) {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            count = Root()->Add(uri.substr(1), v);
        } else {
            auto res = m_table_.emplace(uri.substr(0, pos), New());
            if (res.second) { res.first->second->m_parent_ = Self(); }
            if (pos != std::string::npos) {
                count = res.first->second->Add(uri.substr(pos), v);
            } else {
                //                if (m_entity_ != nullptr) {
                //                    res.first->second->Set("0", m_entity_);
                //                    m_entity_.reset();
                //                }
                //                res.first->second->Set(std::to_string(res.first->second->m_table_.size()), v);
                //                res.first->second->m_node_type_ = DN_ARRAY;
                count = 1;
            }
        }
    }
    return count;
}
std::shared_ptr<const DataNode> DataNodeTableMemory::Get(std::string const& uri) const {
    if (uri.empty()) { return nullptr; }

    std::shared_ptr<const DataNode> res = nullptr;

    auto pos = uri.find(SP_URL_SPLIT_CHAR);
    if (pos == 0) {
        res = Root()->Get(uri.substr(1));
    } else {
        auto it = m_table_.find(uri.substr(0, pos));
        if (it != m_table_.end()) {
            res = pos == std::string::npos ? it->second : it->second->Get(uri.substr(pos + 1));
        }
    }

    return res;
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
    std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const {
    size_type count = 0;
    for (auto const& item : m_table_) { count += f(item.first, item.second); }
    return count;
}

DataNodeMemory::DataNodeMemory() {}
DataNodeMemory::~DataNodeMemory() {}
std::shared_ptr<DataNode> DataNodeMemory::New() { return DataNodeTableMemory::New(); }
std::shared_ptr<DataNodeEntity> DataNodeMemory::NewEntity(std::shared_ptr<DataEntity> const& v) const {
    auto res = DataNodeEntityMemory::New(v);
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNodeTable> DataNodeMemory::NewTable() const {
    auto res = DataNodeTableMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNodeArray> DataNodeMemory::NewArray() const {
    auto res = DataNodeArrayMemory::New();
    res->m_parent_ = Self();
    return res;
}
std::shared_ptr<DataNodeFunction> DataNodeMemory::NewFunction() const {
    auto res = DataNodeFunctionMemory::New();
    res->m_parent_ = Self();
    return res;
}

REGISTER_CREATOR(DataNodeMemory, mem);

}  // namespace data {
}  // namespace simpla{