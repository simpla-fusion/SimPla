//
// Created by salmon on 17-3-6.
//
#include "DataNodeMemory.h"
#include <iomanip>
#include <map>
#include <regex>
#include "../DataBlock.h"
#include "../DataNode.h"
namespace simpla {
namespace data {

REGISTER_CREATOR(DataNodeMemory, mem);

DataNodeMemory::DataNodeMemory() : DataNode(DataNode::DN_TABLE){};
DataNodeMemory::~DataNodeMemory() = default;

size_type DataNodeMemory::size() const { return m_table_.size(); }
size_type DataNodeMemory::Set(index_type s, std::shared_ptr<const DataNode> v) { return Set(std::to_string(s), v); }
size_type DataNodeMemory::Add(index_type s, std::shared_ptr<const DataNode> v) { return Add(std::to_string(s), v); }
size_type DataNodeMemory::Delete(index_type s) { return Delete(std::to_string(s)); }

size_type DataNodeMemory::Add(std::shared_ptr<const DataNode> v) { return Set(std::to_string(size()), v); };
std::shared_ptr<const DataNode> DataNodeMemory::Get(index_type s) const { return Get(std::to_string(s)); }

size_type DataNodeMemory::Set(std::string const& uri, std::shared_ptr<const DataNode> v) {
    if (uri.empty() || v == nullptr) { return 0; }
    if (uri[0] == SP_URL_SPLIT_CHAR) { return Root()->Set(uri.substr(1), v); }

    size_type count = 0;
    auto obj = Self();
    std::string k = uri;
    while (obj != nullptr && !k.empty()) {
        size_type tail = k.find(SP_URL_SPLIT_CHAR);
        if (tail == std::string::npos) {
            obj->m_table_[k] = v->Duplicate();  // insert_or_assign
            count = v->size();
            break;
        } else {
            obj = std::dynamic_pointer_cast<this_type>(
                obj->m_table_.emplace(k.substr(0, tail), CreateNode(DN_TABLE)).first->second);
            k = k.substr(tail + 1);
        }
    }
    return count;
}
size_type DataNodeMemory::Add(std::string const& uri, std::shared_ptr<const DataNode> v) {
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
            if (auto q = std::dynamic_pointer_cast<DataNodeMemory>(obj)) { count = q->Add(v); }
            break;
        } else {
            obj = p->m_table_.emplace(k.substr(0, tail), CreateNode(DN_TABLE)).first->second;
            k = k.substr(tail + 1);
        }
    }
    return count;
}
std::shared_ptr<const DataNode> DataNodeMemory::Get(std::string const& uri) const {
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
            obj = nullptr;  // obj->Get(k.substr(0, tail));
        }
        if (tail != std::string::npos) {
            k = k.substr(tail + 1);
        } else {
            k = "";
        };
    }
    return obj;
};
size_type DataNodeMemory::Delete(std::string const& uri) {
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
size_type DataNodeMemory::Foreach(
    std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& f) const {
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
            res = DataNodeMemory::New(DN_ARRAY);
            break;
        case DN_TABLE:
            res = DataNodeMemory::New(DN_TABLE);
            break;
        case DN_FUNCTION:
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