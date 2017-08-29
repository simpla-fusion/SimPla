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
    eNodeType m_node_type_ = DN_TABLE;
    std::shared_ptr<DataNodeMemory> m_parent_ = nullptr;
    std::map<std::string, std::shared_ptr<DataNodeMemory>> m_table_;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;
};
DataNodeMemory::DataNodeMemory() : m_pimpl_(new pimpl_s) {}
DataNodeMemory::~DataNodeMemory() { delete m_pimpl_; }

// DataNodeMemory::DataNodeMemory(std::shared_ptr<DataEntity> const& v) : DataNodeMemory() {
//    m_pimpl_->m_entity_ = v;
//    m_pimpl_->m_node_type_ = DN_ENTITY;
//};

std::shared_ptr<DataNode> DataNodeMemory::Duplicate() const {
    auto res = DataNodeMemory::New();
    res->m_pimpl_->m_parent_ = m_pimpl_->m_parent_;
    return res;
}
size_type DataNodeMemory::size() const { return m_pimpl_->m_table_.size(); }

/** @addtogroup{ Interface */
DataNode::eNodeType DataNodeMemory::type() const { return m_pimpl_->m_node_type_; }

std::shared_ptr<DataNode> DataNodeMemory::Root() const {
    return m_pimpl_->m_parent_ != nullptr ? m_pimpl_->m_parent_->Root()
                                          : const_cast<this_type*>(this)->shared_from_this();
}
std::shared_ptr<DataNode> DataNodeMemory::Parent() const { return m_pimpl_->m_parent_; }

void DataNodeMemory::Clear() { m_pimpl_->m_table_.clear(); }

std::shared_ptr<DataEntity> DataNodeMemory::GetEntity() const { return m_pimpl_->m_entity_; }

size_type DataNodeMemory::Foreach(
    std::function<size_type(std::string, std::shared_ptr<const DataNode>)> const& fun) const {
    size_type count = 0;
    for (auto const& item : m_pimpl_->m_table_) { count += fun(item.first, item.second); }
    return count;
}
size_type DataNodeMemory::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    size_type count = 0;
    if (!uri.empty()) {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            count = Root()->Set(uri.substr(1), v);
        } else {
            auto p = m_pimpl_->m_table_.emplace(uri.substr(0, pos), New());
            if (p.second) { p.first->second->m_pimpl_->m_parent_ = Self(); }
            if (pos != std::string::npos) {
                count = p.first->second->Set(uri.substr(pos + 1), v);
            } else {
                p.first->second->m_pimpl_->m_table_.clear();
                p.first->second->m_pimpl_->m_entity_ = v;
                p.first->second->m_pimpl_->m_node_type_ = DN_ENTITY;
                count = 1;
            }
        }
    }
    return count;
}
size_type DataNodeMemory::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) {
    return base_type::Add(uri, v);
    //    if (!uri.empty()) {
    //        auto pos = uri.find(SP_URL_SPLIT_CHAR);
    //        if (pos == 0) {
    //            count = Root()->Add(uri.substr(1), v);
    //        } else {
    //            auto res = m_pimpl_->m_table_.emplace(uri.substr(0, pos), New());
    //            if (res.second) { res.first->second->m_pimpl_->m_parent_ = Self(); }
    //            if (pos != std::string::npos) {
    //                count = res.first->second->Add(uri.substr(pos), v);
    //            } else {
    //                if (m_pimpl_->m_entity_ != nullptr) {
    //                    res.first->second->Set("0", m_pimpl_->m_entity_);
    //                    m_pimpl_->m_entity_.reset();
    //                }
    //                res.first->second->Set(std::to_string(res.first->second->m_pimpl_->m_table_.size()), v);
    //                res.first->second->m_pimpl_->m_node_type_ = DN_ARRAY;
    //                count = 1;
    //            }
    //        }
    //    }
}

std::shared_ptr<const DataNode> DataNodeMemory::Get(std::string const& uri) const {
    std::shared_ptr<const DataNode> res = nullptr;

    if (uri.empty()) {
        res = shared_from_this();
    } else {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            res = Root()->Get(uri.substr(1));
        } else {
            auto it = m_pimpl_->m_table_.find(uri.substr(0, pos));
            if (it != m_pimpl_->m_table_.end()) {
                res = pos == std::string::npos ? it->second : it->second->Get(uri.substr(pos + 1));
            }
        }
    }

    return res;
};

size_type DataNodeMemory::Delete(std::string const& uri) {
    size_type count = 0;
    if (uri.empty()) {
    } else {
        auto pos = uri.find(SP_URL_SPLIT_CHAR);
        if (pos == 0) {
            count = Root()->Delete(uri.substr(1));
        } else {
            auto it = m_pimpl_->m_table_.find(uri.substr(0, pos));
            if (it != m_pimpl_->m_table_.end()) {
                if (pos == std::string::npos) {
                    m_pimpl_->m_table_.erase(it);
                    count = 1;
                } else {
                    count = it->second->Delete(uri.substr(pos));
                };
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
//                   (dynamic_cast<data::DataTable const*>(res->m_node_.get()) != nullptr);
//        },
//        [&](DataNodeMemory* s_t, std::string const& k) {
//            return (std::dynamic_pointer_cast<DataNodeMemory>(
//                        std::dynamic_pointer_cast<DataTable>(s_t->m_pimpl_->m_table_.find(k)->m_node_)->database())
//                        .get());
//        },
//        [&](DataNodeMemory* s_t, std::string const& k) -> DataNodeMemory* {
//            if (return_if_not_exist) { return nullptr; }
//            auto res = s_t->m_pimpl_->m_table_.emplace(k, DataTable::New());
//            return std::dynamic_pointer_cast<DataNodeMemory>(
//                       std::dynamic_pointer_cast<DataTable>(res.first->m_node_)->database())
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
// std::shared_ptr<DataEntity> DataNodeMemory::GetEntity(std::string const& url) const {
//    std::shared_ptr<DataEntity> res = nullptr;
//    auto t = m_pimpl_->get_table(const_cast<DataNodeMemory*>(this), url, false);
//    if (t.first != nullptr && !t.m_node_.empty()) {
//        auto it = t.first->m_pimpl_->m_table_.find(t.m_node_);
//        if (it != t.first->m_pimpl_->m_table_.end()) { res = it->m_node_; }
//    }
//
//    return res;
//};
//
// int DataNodeMemory::SetEntity(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    auto tab_res = pimpl_s::get_table((this), uri, true);
//    if (tab_res.m_node_.empty() || tab_res.first == nullptr) { return 0; }
//    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.m_node_, nullptr);
//    if (res.first->m_node_ == nullptr) { res.first->m_node_ = v; }
//
//    size_type count = 1;
//    auto dst = std::dynamic_pointer_cast<DataTable>(GetEntity(uri));
//    auto src = std::dynamic_pointer_cast<DataTable>(p);
//
//    if (dst != nullptr && src != nullptr) {
//        dst->SetTable(*src);
//        count = src->Count();
//    } else {
//        SetEntity(uri, p);
//    }
//    return 1;
//}
// int DataNodeMemory::AddEntity(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
//    auto tab_res = pimpl_s::get_table(this, uri, false);
//    if (tab_res.m_node_.empty()) { return 0; }
//    auto res = tab_res.first->m_pimpl_->m_table_.emplace(tab_res.m_node_, DataArray::New());
//    if (dynamic_cast<DataArray const*>(res.first->m_node_.get()) != nullptr &&
//        res.first->m_node_->value_type_info() == v->value_type_info()) {
//    } else if (std::dynamic_pointer_cast<DataArray>(res.first->m_node_) != nullptr) {
//        auto t_array = DataArray::New();
//        t_array->AddEntity(res.first->m_node_);
//        res.first->m_node_ = t_array;
//    }
//    std::dynamic_pointer_cast<DataArray>(res.first->m_node_)->AddEntity(v);
//
//    return 1;
//}
// int DataNodeMemory::Delete(std::string const& uri) {
//    auto res = m_pimpl_->get_table(this, uri, true);
//    return (res.first->m_pimpl_->m_table_.erase(res.m_node_));
//}
//
// int DataNodeMemory::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
//    int counter = 0;
//    for (auto const& item : m_pimpl_->m_table_) { counter += f(item.first, item.m_node_); }
//    return counter;
//}

}  // namespace data {
}  // namespace simpla{