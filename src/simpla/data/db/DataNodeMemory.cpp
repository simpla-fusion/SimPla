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
REGISTER_CREATOR(DataNodeMemory, mem);

struct DataNodeMemory::pimpl_s {
    e_NodeType m_node_type = DN_NULL;
    std::shared_ptr<DataNodeMemory> m_parent_ = nullptr;
    std::map<std::string, std::shared_ptr<DataNodeMemory>> m_table_;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
};
DataNodeMemory::DataNodeMemory() : m_pimpl_(new pimpl_s) {}
DataNodeMemory::~DataNodeMemory() { delete m_pimpl_; }

DataNodeMemory::DataNodeMemory(std::shared_ptr<DataNodeMemory> const& v) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_parent_ = v;
};
int DataNodeMemory::Connect(std::string const& authority, std::string const& path, std::string const& query,
                            std::string const& fragment) {
    return 0;
}
int DataNodeMemory::Disconnect() { return 0; }
bool DataNodeMemory::isValid() const { return true; }
int DataNodeMemory::Flush() { return 0; }

std::shared_ptr<DataNode> DataNodeMemory::Duplicate() const { return DataNodeMemory::New(m_pimpl_->m_parent_); }
size_type DataNodeMemory::GetNumberOfChildren() const { return m_pimpl_->m_table_.size(); }

/** @addtogroup{ Interface */
DataNode::e_NodeType DataNodeMemory::NodeType() const { return m_pimpl_->m_node_type; }

std::shared_ptr<DataNode> DataNodeMemory::Root() {
    return m_pimpl_->m_parent_ != nullptr ? m_pimpl_->m_parent_->Root() : shared_from_this();
}
std::shared_ptr<DataNode> DataNodeMemory::Parent() const { return m_pimpl_->m_parent_; }

void DataNodeMemory::Clear() { m_pimpl_->m_table_.clear(); }

std::shared_ptr<DataEntity> DataNodeMemory::Get() { return m_pimpl_->m_entity_; }
std::shared_ptr<DataEntity> DataNodeMemory::Get() const { return m_pimpl_->m_entity_; }

int DataNodeMemory::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) {
    int count = 0;
    for (auto& item : m_pimpl_->m_table_) { count += fun(item.first, item.second); }
    return 0;
}
int DataNodeMemory::Foreach(std::function<int(std::string, std::shared_ptr<DataNode>)> const& fun) const {
    int count = 0;
    for (auto const& item : m_pimpl_->m_table_) { count += fun(item.first, item.second); }
    return 0;
}
int DataNodeMemory::Set(std::shared_ptr<DataEntity> const& v) {
    if (!m_pimpl_->m_table_.empty()) { RUNTIME_ERROR << "Can not insert entity to Table/Array node!" << std::endl; }
    m_pimpl_->m_node_type = DN_ENTITY;
    m_pimpl_->m_entity_ = v;
    return 1;
}
int DataNodeMemory::Add(std::shared_ptr<DataEntity> const& v) { return AddNode()->Set(v); }

std::shared_ptr<DataNode> DataNodeMemory::GetNode(std::string const& s, int flag) {
    std::shared_ptr<DataNode> res = nullptr;
    std::string uri = s;
    if ((flag & RECURSIVE) == 0) {
        if (m_pimpl_->m_entity_ != nullptr) {
            auto n = std::dynamic_pointer_cast<DataNodeMemory>(Duplicate());
            n->Set(m_pimpl_->m_entity_);
            m_pimpl_->m_entity_.reset();
            if ((flag & ADD_IF_EXIST) != 0) {
                m_pimpl_->m_table_["0"] = n;
                m_pimpl_->m_node_type = DN_ARRAY;
                if (uri == "0") { uri = std::to_string(1); }
            } else {
                m_pimpl_->m_table_["_"] = n;
                m_pimpl_->m_node_type = DN_TABLE;
            }
        }
        if ((flag & NEW_IF_NOT_EXIST) != 0) {
            auto r = m_pimpl_->m_table_.emplace(uri, std::dynamic_pointer_cast<DataNodeMemory>(Duplicate()));
            res = r.first->second;
            if (r.second) {
                if (((flag & ADD_IF_EXIST) != 0) &&
                    (m_pimpl_->m_node_type == DN_NULL || m_pimpl_->m_node_type == DN_ARRAY)) {
                    m_pimpl_->m_node_type = DN_ARRAY;
                } else {
                    m_pimpl_->m_node_type = DN_TABLE;
                }
            }
        } else {
            auto r = m_pimpl_->m_table_.find(uri);
            res = r == m_pimpl_->m_table_.end() ? DataNode::New() : r->second;
        }
    } else {
        res = RecursiveFindNode(shared_from_this(), uri, flag).second;
    }

    return res;
};

std::shared_ptr<DataNode> DataNodeMemory::GetNode(index_type s, int flag) {
    return GetNode(std::to_string(s), flag | ADD_IF_EXIST);
}
std::shared_ptr<DataNode> DataNodeMemory::GetNode(index_type s, int flag) const {
    return GetNode(std::to_string(s), flag);
}
std::shared_ptr<DataNode> DataNodeMemory::GetNode(std::string const& uri, int flag) const {
    std::shared_ptr<DataNode> res = nullptr;
    if ((flag & RECURSIVE) == 0) {
        auto it = m_pimpl_->m_table_.find(uri);
        res = (it == m_pimpl_->m_table_.end()) ? DataNode::New() : it->second;
    } else {
        res = RecursiveFindNode(const_cast<this_type*>(this)->shared_from_this(), uri, flag).second;
    }
    return res;
};
int DataNodeMemory::DeleteNode(std::string const& uri, int flag) {
    int count = 0;
    if ((flag & RECURSIVE) == 0) {
        count = static_cast<int>(m_pimpl_->m_table_.erase(uri));
    } else {
        auto node = RecursiveFindNode(shared_from_this(), uri, RECURSIVE);
        while (node.second != nullptr) {
            if (auto p = std::dynamic_pointer_cast<DataNodeMemory>(node.second->Parent())) {
                count += static_cast<int>(p->m_pimpl_->m_table_.erase(node.first));
                node.second = p;
            }
        }
    }
    return count;
}

//
// struct DataNodeMemory::pimpl_s {
//    typedef std::map<std::string, std::shared_ptr<DataEntity>> table_type;
//    table_type m_table_;
//    static std::pair<DataNodeMemory*, std::string> get_table(DataNodeMemory* self, std::string const& uri,
//                                                             bool return_if_not_exist);
//};
//
// std::pair<DataNodeMemory*, std::string> DataNodeMemory::pimpl_s::get_table(DataNodeMemory* t, std::string const&
// uri,
//                                                                           bool return_if_not_exist) {
//    return HierarchicalTableForeach(
//        t, uri,
//        [&](DataNodeMemory* s_t, std::string const& k) -> bool {
//            auto res = s_t->m_pimpl_->m_table_.find(k);
//            return (res != s_t->m_pimpl_->m_table_.end()) &&
//                   (dynamic_cast<data::DataTable const*>(res->second.get()) != nullptr);
//        },
//        [&](DataNodeMemory* s_t, std::string const& k) {
//            return (std::dynamic_pointer_cast<DataNodeMemory>(
//                        std::dynamic_pointer_cast<DataTable>(s_t->m_pimpl_->m_table_.find(k)->second)->database())
//                        .get());
//        },
//        [&](DataNodeMemory* s_t, std::string const& k) -> DataNodeMemory* {
//            if (return_if_not_exist) { return nullptr; }
//            auto res = s_t->m_pimpl_->m_table_.emplace(k, DataTable::New());
//            return std::dynamic_pointer_cast<DataNodeMemory>(
//                       std::dynamic_pointer_cast<DataTable>(res.first->second)->database())
//                .get();
//
//        });
//};
//
// DataNodeMemory::DataNodeMemory() : m_pimpl_(new pimpl_s) {}
// DataNodeMemory::DataNodeMemory() { delete m_pimpl_; };
//
// int DataNodeMemory::Connect(std::string const& authority, std::string const& path, std::string const& query,
//                            std::string const& fragment) {
//    return SP_SUCCESS;
//};
// int DataNodeMemory::Disconnect() { return SP_SUCCESS; };
// int DataNodeMemory::Flush() { return SP_SUCCESS; };
//
//// std::ostream& DataBaseMemory::Print(std::ostream& os, int indent) const { return os; };
//
// bool DataNodeMemory::isNull() const { return m_pimpl_ == nullptr; };
//
// std::shared_ptr<DataEntity> DataNodeMemory::Get(std::string const& url) const {
//    std::shared_ptr<DataEntity> res = nullptr;
//    auto t = m_pimpl_->get_table(const_cast<DataNodeMemory*>(this), url, false);
//    if (t.first != nullptr && !t.second.empty()) {
//        auto it = t.first->m_pimpl_->m_table_.find(t.second);
//        if (it != t.first->m_pimpl_->m_table_.end()) { res = it->second; }
//    }
//
//    return res;
//};
//
// int DataNodeMemory::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    auto tab_res = pimpl_s::get_table((this), uri, true);
//    if (tab_res.second.empty() || tab_res.first == nullptr) { return 0; }
//    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, nullptr);
//    if (res.first->second == nullptr) { res.first->second = v; }
//
//    size_type count = 1;
//    auto dst = std::dynamic_pointer_cast<DataTable>(Get(uri));
//    auto src = std::dynamic_pointer_cast<DataTable>(p);
//
//    if (dst != nullptr && src != nullptr) {
//        dst->SetTable(*src);
//        count = src->Count();
//    } else {
//        Set(uri, p);
//    }
//    return 1;
//}
// int DataNodeMemory::Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    auto tab_res = pimpl_s::get_table(this, uri, false);
//    if (tab_res.second.empty()) { return 0; }
//    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.second, DataArray::New());
//    if (dynamic_cast<DataArray const*>(res.first->second.get()) != nullptr &&
//        res.first->second->value_type_info() == v->value_type_info()) {
//    } else if (std::dynamic_pointer_cast<DataArray>(res.first->second) != nullptr) {
//        auto t_array = DataArray::New();
//        t_array->Add(res.first->second);
//        res.first->second = t_array;
//    }
//    std::dynamic_pointer_cast<DataArray>(res.first->second)->Add(v);
//
//    return 1;
//}
// int DataNodeMemory::Delete(std::string const& uri) {
//    auto res = m_pimpl_->get_table(this, uri, true);
//    return (res.first->m_pimpl_->m_table_.erase(res.second));
//}
//
// int DataNodeMemory::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
//    int counter = 0;
//    for (auto const& item : m_pimpl_->m_table_) { counter += f(item.first, item.second); }
//    return counter;
//}

}  // namespace data {
}  // namespace simpla{