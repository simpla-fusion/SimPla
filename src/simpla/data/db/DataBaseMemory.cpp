//
// Created by salmon on 17-3-6.
//
#include "DataBaseMemory.h"
#include <iomanip>
#include <map>
#include <regex>
#include "../DataBlock.h"
#include "../DataEntity.h"
#include "../DataNode.h"
#include "DataUtility.h"

namespace simpla {
namespace data {
REGISTER_CREATOR(DataBaseMemory, mem);

struct DataBaseMemory::Node : public DataNode {
    SP_DEFINE_FANCY_TYPE_NAME(Node, DataNode)

    std::shared_ptr<Node> m_parent_ = nullptr;
    std::shared_ptr<Node> m_root_ = nullptr;

    std::map<std::string, std::shared_ptr<Node>> m_table_;
    typedef std::map<std::string, std::shared_ptr<Node>>::iterator iterator;
    iterator m_it_, m_end_;
    std::shared_ptr<DataEntity> m_entity_ = nullptr;

   protected:
    Node() = default;
    Node(std::shared_ptr<Node> const& parent, iterator b, iterator e)
        : m_parent_(std::move(parent)), m_it_(b), m_end_(e) {
        m_root_ = std::dynamic_pointer_cast<this_type>(m_parent_.get() == nullptr ? this->shared_from_this()
                                                                                  : m_parent_->Root());
        if (m_it_->second != nullptr) { m_entity_ = m_it_->second->m_entity_; }
    };
    Node(std::shared_ptr<DataEntity> const& v) : m_entity_(v){};
    //    Node(std::shared_ptr<DataNode> const& parent, const iterator& b, const iterator& e)
    //        : m_parent_(std::dynamic_pointer_cast<this_type>(parent)), m_it_(b), m_end_(e) {
    //        m_root_ = std::dynamic_pointer_cast<this_type>(m_parent_.get() == nullptr ? this->shared_from_this()
    //                                                                                  : m_parent_->Root());
    //        m_entity_ = m_it_->second;
    //    };

   public:
    ~Node() override = default;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args&&... args) {
        return std::shared_ptr<Node>(new Node(std::forward<Args>(args)...));
    }
    std::shared_ptr<DataNode> Duplicate() const override { return Node::New(m_parent_, m_it_, m_end_); }

    /** @addtogroup{ Interface */
    int Flush() override { return 0; }
    bool isNull() const override { return m_table_.empty() && m_it_ == m_end_; }
    bool isArray() const override { return false; }
    bool isTable() const override { return !m_table_.empty(); }
    bool isEntity() const override { return m_it_ != m_end_; }
    size_type GetNumberOfChildren() const override { return m_table_.size(); }

    std::shared_ptr<DataNode> Root() override { return m_root_; }
    std::shared_ptr<DataNode> Parent() const override { return m_parent_; }
    std::shared_ptr<DataNode> FirstChild() const override {
        auto self = std::dynamic_pointer_cast<this_type>(const_cast<this_type*>(this)->shared_from_this());
        return m_table_.empty() ? New() : New(self, self->m_table_.begin(), self->m_table_.end());
    }
    std::shared_ptr<DataNode> Next() const override {
        iterator it = m_it_;
        ++it;
        return m_it_ == m_end_ ? nullptr : New(m_parent_, it, m_end_);
    }

    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) override;
    std::shared_ptr<DataNode> GetNode(std::string const& uri, int flag) const override;

    int DeleteNode(std::string const& uri, int flag) override;

    std::string GetKey() const override { return "KEY"; /*m_it_ == m_end_ ? "KEY" : m_it_->first;*/ }
    std::shared_ptr<DataEntity> GetEntity() override { return m_entity_; }
    std::shared_ptr<DataEntity> GetEntity() const override { return m_entity_; }
    int SetEntity(std::shared_ptr<DataEntity> const& v) override {
        m_entity_ = v;
        return 1;
    }
    /** @} */
};
std::shared_ptr<DataNode> DataBaseMemory::Node::GetNode(std::string const& uri, int flag) {
    std::shared_ptr<DataNode> res = nullptr;

    //    if ((flag & RECURSIVE) == 0) {
    //        if ((flag & NEW_IF_NOT_EXIST) != 0) {
    auto r = m_table_.emplace(uri, New());
    if (r.second) {
        r.first->second->m_parent_ = std::dynamic_pointer_cast<this_type>(shared_from_this());
        r.first->second->m_it_ = r.first;
        r.first->second->m_end_ = m_table_.end();
    }
    res = r.first->second;
    //        }
    //    } else {
    //        res = RecursiveFindNode(shared_from_this(), uri, flag).first;
    //    }
    return res;
};
std::shared_ptr<DataNode> DataBaseMemory::Node::GetNode(std::string const& uri, int flag) const {
    std::shared_ptr<DataNode> res = nullptr;
    //    if ((flag & RECURSIVE) == 0) {
    auto it = m_table_.find(uri);
    res = (it == m_table_.end()) ? DataNode::New() : it->second;
    //    } else {
    //        res = RecursiveFindNode(const_cast<this_type*>(this)->shared_from_this(), uri, flag).first;
    //    }
    return res;
};
int DataBaseMemory::Node::DeleteNode(std::string const& uri, int flag) {
    int count = 0;
    if ((flag & RECURSIVE) == 0) {
        count = static_cast<int>(m_table_.erase(uri));
    } else {
        auto node = RecursiveFindNode(shared_from_this(), uri, flag).first;
        while (node != nullptr) {
            if (auto p = std::dynamic_pointer_cast<Node>(node->Parent())) {
                count += static_cast<int>(p->m_table_.erase(node->GetKey()));
                node = p;
            }
        }
    }
    return count;
}

DataBaseMemory::DataBaseMemory() {}
DataBaseMemory::~DataBaseMemory() {}
int DataBaseMemory::Connect(std::string const& authority, std::string const& path, std::string const& query,
                            std::string const& fragment) {
    return 0;
}
int DataBaseMemory::Disconnect() { return 0; }
bool DataBaseMemory::isNull() const { return false; }
int DataBaseMemory::Flush() { return 0; }
std::shared_ptr<DataNode> DataBaseMemory::Root() { return Node::New(); }
//
// struct DataBaseMemory::pimpl_s {
//    typedef std::map<std::string, std::shared_ptr<DataEntity>> table_type;
//    table_type m_table_;
//    static std::pair<DataBaseMemory*, std::string> get_table(DataBaseMemory* self, std::string const& uri,
//                                                             bool return_if_not_exist);
//};
//
// std::pair<DataBaseMemory*, std::string> DataBaseMemory::pimpl_s::get_table(DataBaseMemory* t, std::string const& uri,
//                                                                           bool return_if_not_exist) {
//    return HierarchicalTableForeach(
//        t, uri,
//        [&](DataBaseMemory* s_t, std::string const& k) -> bool {
//            auto res = s_t->m_pimpl_->m_table_.find(k);
//            return (res != s_t->m_pimpl_->m_table_.end()) &&
//                   (dynamic_cast<data::DataTable const*>(res->second.get()) != nullptr);
//        },
//        [&](DataBaseMemory* s_t, std::string const& k) {
//            return (std::dynamic_pointer_cast<DataBaseMemory>(
//                        std::dynamic_pointer_cast<DataTable>(s_t->m_pimpl_->m_table_.find(k)->second)->database())
//                        .get());
//        },
//        [&](DataBaseMemory* s_t, std::string const& k) -> DataBaseMemory* {
//            if (return_if_not_exist) { return nullptr; }
//            auto res = s_t->m_pimpl_->m_table_.emplace(k, DataTable::New());
//            return std::dynamic_pointer_cast<DataBaseMemory>(
//                       std::dynamic_pointer_cast<DataTable>(res.first->second)->database())
//                .get();
//
//        });
//};
//
// DataBaseMemory::DataBaseMemory() : m_pimpl_(new pimpl_s) {}
// DataBaseMemory::~DataBaseMemory() { delete m_pimpl_; };
//
// int DataBaseMemory::Connect(std::string const& authority, std::string const& path, std::string const& query,
//                            std::string const& fragment) {
//    return SP_SUCCESS;
//};
// int DataBaseMemory::Disconnect() { return SP_SUCCESS; };
// int DataBaseMemory::Flush() { return SP_SUCCESS; };
//
//// std::ostream& DataBaseMemory::Print(std::ostream& os, int indent) const { return os; };
//
// bool DataBaseMemory::isNull() const { return m_pimpl_ == nullptr; };
//
// std::shared_ptr<DataEntity> DataBaseMemory::Get(std::string const& url) const {
//    std::shared_ptr<DataEntity> res = nullptr;
//    auto t = m_pimpl_->get_table(const_cast<DataBaseMemory*>(this), url, false);
//    if (t.first != nullptr && !t.second.empty()) {
//        auto it = t.first->m_pimpl_->m_table_.find(t.second);
//        if (it != t.first->m_pimpl_->m_table_.end()) { res = it->second; }
//    }
//
//    return res;
//};
//
// int DataBaseMemory::Set(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
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
// int DataBaseMemory::Add(std::string const& uri, const std::shared_ptr<DataEntity>& v) {
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
// int DataBaseMemory::Delete(std::string const& uri) {
//    auto res = m_pimpl_->get_table(this, uri, true);
//    return (res.first->m_pimpl_->m_table_.erase(res.second));
//}
//
// int DataBaseMemory::Foreach(std::function<int(std::string const&, std::shared_ptr<DataEntity>)> const& f) const {
//    int counter = 0;
//    for (auto const& item : m_pimpl_->m_table_) { counter += f(item.first, item.second); }
//    return counter;
//}

}  // namespace data {
}  // namespace simpla{