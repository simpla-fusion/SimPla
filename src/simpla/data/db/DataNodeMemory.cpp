//
// Created by salmon on 17-3-6.
//
#include <iomanip>
#include <map>
#include <regex>
#include "../DataBlock.h"
#include "../DataNode.h"
namespace simpla {
namespace data {
struct DataNodeMemory : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemory, DataNode);
    SP_DATA_NODE_HEAD(DataNodeMemory)

   protected:
    explicit DataNodeMemory(DataNode::eNodeType etype);

   public:
    std::shared_ptr<DataNode> CreateNode(eNodeType e_type) const override;
};
REGISTER_CREATOR(DataNodeMemory, mem);

DataNodeMemory::DataNodeMemory() : DataNode(DN_NULL) {}
DataNodeMemory::DataNodeMemory(DataNode::eNodeType etype) : DataNode(etype) {}
DataNodeMemory::~DataNodeMemory() = default;

struct DataNodeMemoryFunction : public DataNodeMemory {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemoryFunction, DataNodeMemory);
    SP_DATA_NODE_HEAD(DataNodeMemoryFunction)
};
DataNodeMemoryFunction::DataNodeMemoryFunction() : DataNodeMemory(DataNode::DN_FUNCTION) {}
DataNodeMemoryFunction::~DataNodeMemoryFunction() = default;

struct DataNodeMemoryArray : public DataNodeMemory {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemoryArray, DataNodeMemory)
    SP_DATA_NODE_HEAD(DataNodeMemoryArray)

   public:
    size_type size() const override;
    size_type Set(size_type s, std::shared_ptr<DataNode> const& v) override;
    size_type Add(size_type s, std::shared_ptr<DataNode> const& v) override;
    size_type Delete(size_type s) override;
    std::shared_ptr<DataNode> Get(size_type s) const override;
    std::shared_ptr<DataNode> Get(std::string const& s) const override;
    size_type Add(std::shared_ptr<DataNode> const& v) override;

   private:
    std::vector<std::shared_ptr<DataNode>> m_data_;
};

DataNodeMemoryArray::DataNodeMemoryArray() : DataNodeMemory(DataNode::DN_ARRAY){};
DataNodeMemoryArray::~DataNodeMemoryArray() = default;

size_type DataNodeMemoryArray::size() const { return m_data_.size(); }
size_type DataNodeMemoryArray::Set(size_type s, std::shared_ptr<DataNode> const& v) {
    if (s >= size()) { m_data_.resize(s + 1); }
    m_data_[s] = (v);
    return 1;
}
size_type DataNodeMemoryArray::Add(size_type s, std::shared_ptr<DataNode> const& v) {
    if (s >= size()) { return 0; }
    if (auto p = std::dynamic_pointer_cast<DataNodeMemoryArray>(m_data_[s])) { return p->Set(p->size(), v); }
    return 0;
}
size_type DataNodeMemoryArray::Delete(size_type s) {
    FIXME;
    return 0;
}

size_type DataNodeMemoryArray::Add(std::shared_ptr<DataNode> const& v) {
    m_data_.push_back(v);
    return 1;
};

std::shared_ptr<DataNode> DataNodeMemoryArray::Get(size_type s) const { return m_data_.at(s); }
std::shared_ptr<DataNode> DataNodeMemoryArray::Get(std::string const& s) const {
    return Get(static_cast<size_type>(std::stoi(s, nullptr, 10)));
};

struct DataNodeMemoryTable : public DataNodeMemory {
    SP_DEFINE_FANCY_TYPE_NAME(DataNodeMemoryTable, DataNodeMemory)
    SP_DATA_NODE_HEAD(DataNodeMemoryTable)

   public:
    size_type size() const override;

    size_type Set(std::string const& uri, std::shared_ptr<DataNode> const& v) override;
    size_type Add(std::string const& uri, std::shared_ptr<DataNode> const& v) override;
    size_type Delete(std::string const& uri) override;
    std::shared_ptr<DataNode> Get(std::string const& uri) const override;
    size_type Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const override;

   private:
    std::map<std::string, std::shared_ptr<DataNode>> m_table_;
};
DataNodeMemoryTable::DataNodeMemoryTable() : DataNodeMemory(DN_TABLE) {}
DataNodeMemoryTable::~DataNodeMemoryTable() = default;
size_type DataNodeMemoryTable::size() const { return m_table_.size(); }
size_type DataNodeMemoryTable::Set(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    auto obj = Self();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {
        size_type tail = k.find(SP_URL_SPLIT_CHAR);
        if (tail == std::string::npos) {
            obj->m_table_[k] = v;  // insert_or_assign
            count = v->size();
            break;
        } else {
            obj = std::dynamic_pointer_cast<DataNodeMemoryTable>(
                obj->m_table_.emplace(k.substr(0, tail), CreateNode(DN_TABLE)).first->second);
            k = k.substr(tail + 1);
        }
    }
    return count;
}
size_type DataNodeMemoryTable::Add(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    size_type tail = 0;
    auto obj = shared_from_this();
    std::string k = uri;
    while (obj != nullptr) {
        auto p = std::dynamic_pointer_cast<this_type>(obj);
        if (p == nullptr) { break; }
        tail = k.find(SP_URL_SPLIT_CHAR);

        if (tail == std::string::npos) {
            obj = p->m_table_.emplace(k, CreateNode(DN_ARRAY)).first->second;
            if (auto q = std::dynamic_pointer_cast<DataNodeMemoryArray>(obj)) { count = q->Add(v); }
            break;
        } else {
            obj = p->m_table_.emplace(k.substr(0, tail), CreateNode(DN_TABLE)).first->second;
            k = k.substr(tail + 1);
        }
    }
    return count;
}
std::shared_ptr<DataNode> DataNodeMemoryTable::Get(std::string const& uri) const {
    if (uri.empty()) { return nullptr; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Get(uri.substr(1)); }

    auto obj = const_cast<this_type*>(this)->shared_from_this();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {
        auto tail = k.find(SP_URL_SPLIT_CHAR);
        if (auto p = std::dynamic_pointer_cast<this_type>(obj)) {
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
size_type DataNodeMemoryTable::Delete(std::string const& uri) {
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
size_type DataNodeMemoryTable::Foreach(
    std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const {
    size_type count = 0;
    for (auto const& item : m_table_) { count += f(item.first, item.second); }
    return count;
}

std::shared_ptr<DataNode> DataNodeMemory::CreateNode(eNodeType e_type) const {
    std::shared_ptr<DataNode> res = nullptr;
    switch (e_type) {
        case DN_ENTITY:
            res = DataNode::New();
            break;
        case DN_ARRAY:
            res = DataNodeMemoryArray::New();
            break;
        case DN_TABLE:
            res = DataNodeMemoryTable::New();
            break;
        case DN_FUNCTION:
            res = DataNodeMemoryFunction::New();
            break;
        case DN_NULL:
        default:
            break;
    }
    res->SetParent(Self());
    return res;
};

}  // namespace data {
}  // namespace simpla{