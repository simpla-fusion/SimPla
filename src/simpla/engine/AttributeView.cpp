//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <set>
#include <typeindex>
#include "DataBlock.h"
#include "DomainView.h"
#include "MeshView.h"

namespace simpla {
namespace engine {
struct AttributeDict::pimpl_s {
    std::map<id_type, std::shared_ptr<AttributeDesc>> m_db_;
    std::map<std::string, id_type> m_name_map_;
};
AttributeDict::AttributeDict() : m_pimpl_(new pimpl_s) {}
AttributeDict::~AttributeDict() {}
std::ostream &AttributeDict::Print(std::ostream &os, int indent) const {
    for (auto const &item : m_pimpl_->m_db_) {
        os << std::setw(indent + 1) << " " << item.second->name() << " = {"
           << " iform = " << item.second->iform() << ", dof = " << item.second->dof()
           << ", tag = " << item.second->tag() << ", value type = \"" << item.second->value_type_info().name()
           << "\" , config= " << item.second->db() << "  }," << std::endl;
    }
}
id_type AttributeDict::GetGUID(std::string const &s) const { return m_pimpl_->m_name_map_.at(s); }
bool AttributeDict::has(id_type id) const { return m_pimpl_->m_db_.find(id) != m_pimpl_->m_db_.end(); }
bool AttributeDict::has(std::string const &s) const {
    return m_pimpl_->m_name_map_.find(s) != m_pimpl_->m_name_map_.end();
}
std::shared_ptr<AttributeDesc> AttributeDict::Get(id_type id) const { return m_pimpl_->m_db_.at(id); }
std::shared_ptr<AttributeDesc> AttributeDict::Get(std::string const &s) const { return Get(GetGUID(s)); }
std::shared_ptr<AttributeDesc> AttributeDict::Set(id_type gid, std::shared_ptr<AttributeDesc> p) {
    if (p->tag() == SCRATCH) { return nullptr; }
    auto res = m_pimpl_->m_db_.emplace(gid, p);
    if (res.second) {
        m_pimpl_->m_name_map_.emplace(p->name(), gid);
    } else {
        res.first->second->db().merge(p->db());
    }
    return res.first->second;
}

void AttributeDict::Remove(id_type id) {
    m_pimpl_->m_name_map_.erase(Get(id)->name());
    m_pimpl_->m_db_.erase(id);
}
void AttributeDict::Remove(const std::string &s) {
    m_pimpl_->m_db_.erase(GetGUID(s));
    m_pimpl_->m_name_map_.erase(s);
}

void AttributeDict::for_each(std::function<void(AttributeDesc *)> const &fun) {
    for (auto &item : m_pimpl_->m_db_) { fun(item.second.get()); }
}
void AttributeDict::for_each(std::function<void(AttributeDesc const *)> const &fun) const {
    for (auto const &item : m_pimpl_->m_db_) { fun(item.second.get()); }
}
struct AttributeViewBundle::pimpl_s {
    DomainView *m_domain_ = nullptr;
    mutable std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle() : m_pimpl_(new pimpl_s) {}
AttributeViewBundle::~AttributeViewBundle() { OnDestroy(); }
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { os << attr->description().name() << " , "; }
    return os;
};

void AttributeViewBundle::OnNotify() {
    for (auto *item : m_pimpl_->m_attr_views_) { item->OnNotify(); }
}

void AttributeViewBundle::Attach(AttributeView *p) {
    if (p != nullptr && m_pimpl_->m_attr_views_.emplace(p).second) {
        p->Connect(this);
        Click();
    }
}

// void AttributeViewBundle::Detach(AttributeView *p) {
//    if (p != nullptr && m_pimpl_->m_attr_views_.erase(p) > 0) {
//        p->Disconnect(this);
//        Click();
//    }
//}

bool AttributeViewBundle::isModified() {
    return SPObject::isModified() || (m_pimpl_->m_domain_ != nullptr && m_pimpl_->m_domain_->isModified());
}

bool AttributeViewBundle::Update() { return SPObject::Update(); }

// void AttributeViewBundle::RegisterAttribute(AttributeDict *dbase) {
//    for (auto &attr : m_pimpl_->m_attr_views_) { attr->RegisterDescription(dbase); }
//}
DomainView const &AttributeViewBundle::GetDomain() const { return *m_pimpl_->m_domain_; }
MeshView const &AttributeViewBundle::GetMesh() const { return m_pimpl_->m_domain_->GetMesh(); }
std::shared_ptr<DataBlock> &AttributeViewBundle::GetDataBlock(id_type guid) const {
    return m_pimpl_->m_domain_->GetDataBlock(guid);
}

void AttributeViewBundle::Accept(std::function<void(AttributeView *)> const &fun) const {
    for (auto *attr : m_pimpl_->m_attr_views_) { fun(attr); }
}

id_type AttributeDesc::GenerateGUID(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF,
                                    AttributeTag tag) {
    std::string str = name_s + '.' + t_id.name() + '.' + static_cast<char>(IFORM + '0') + '.' +
                      static_cast<char>(DOF + '0') + '.' + static_cast<char>(tag + '0');
    return static_cast<id_type>(std::hash<std::string>{}(str));
}

AttributeDesc::AttributeDesc(const std::string &name_s, const std::type_info &t_id, int IFORM, int DOF,
                             AttributeTag tag)
    : m_name_(name_s),
      m_value_type_info_(t_id),
      m_iform_(IFORM),
      m_dof_(DOF),
      m_tag_(tag),
      m_GUID_(GenerateGUID(name_s, t_id, IFORM, DOF, tag)) {}

AttributeDesc::~AttributeDesc() {}

struct AttributeView::pimpl_s {
    AttributeViewBundle *m_bundle_;
    MeshView const *m_mesh_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    mutable std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    mutable std::shared_ptr<DataBlock> m_data_ = nullptr;
};
AttributeView::AttributeView() : m_pimpl_(new pimpl_s) {}
AttributeView::AttributeView(AttributeViewBundle *b) : AttributeView() { Connect(b); };
AttributeView::AttributeView(MeshView const *m) : AttributeView(){};
AttributeView::~AttributeView() {}

void AttributeView::Config(std::string const &s, AttributeTag t) {
    m_pimpl_->m_desc_ = std::make_shared<AttributeDesc>(s, value_type_info(), iform(), dof(), t);
}

AttributeDesc &AttributeView::description() const {
    if (m_pimpl_->m_desc_ == nullptr) {
        m_pimpl_->m_desc_ = std::make_shared<AttributeDesc>("unnamed", value_type_info(), iform(), dof(), SCRATCH);
    }
    return *m_pimpl_->m_desc_;
}
id_type AttributeView::GUID() const { return description().GUID(); }
std::string const &AttributeView::name() const { return description().name(); }
std::type_info const &AttributeView::value_type_info() const { return description().value_type_info(); }
int AttributeView::iform() const { return description().iform(); }
int AttributeView::dof() const { return description().dof(); }
AttributeTag AttributeView::tag() const { return description().tag(); }
data::DataTable &AttributeView::db() const { return description().db(); }

void AttributeView::Connect(AttributeViewBundle *b) {
    b->OnChanged.Connect<AttributeView, &AttributeView::OnNotify>(this);
    m_pimpl_->m_bundle_ = b;
}

void AttributeView::OnNotify() {
    if (m_pimpl_->m_bundle_ != nullptr) {
        m_pimpl_->m_mesh_ = &m_pimpl_->m_bundle_->GetMesh();
        m_pimpl_->m_data_ = m_pimpl_->m_bundle_->GetDataBlock(GUID());
    } else {
        DO_NOTHING;
    }
}

MeshView const &AttributeView::GetMesh() const {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    return *m_pimpl_->m_mesh_;
};

DataBlock &AttributeView::GetDataBlock() const {
    if (m_pimpl_->m_data_ == nullptr) { m_pimpl_->m_data_ = std::make_shared<DataBlock>(); };
    return *m_pimpl_->m_data_;
}

void AttributeView::InitializeData(){};

/**
 * @startuml
 * start
 *  if (m_domain_ == nullptr) then (yes)
 *  else   (no)
 *    : m_current_block_id = m_domain-> current_block_id();
 *  endif
 * stop
 * @enduml
 */
bool AttributeView::Update() { return SPObject::Update(); }

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }

std::ostream &AttributeView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " " << description().name();
    return os;
};

}  //{ namespace engine
}  // namespace simpla
