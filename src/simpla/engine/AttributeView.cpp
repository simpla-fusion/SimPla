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

struct AttributeViewBundle::pimpl_s {
    DomainView *m_domain_ = nullptr;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle(std::shared_ptr<data::DataTable> const &t)
    : SPObject(t), m_pimpl_(new pimpl_s) {}
AttributeViewBundle::~AttributeViewBundle() {}
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { os << attr->name() << " , "; }
    return os;
};

void AttributeViewBundle::OnNotify() {
    for (auto *item : m_pimpl_->m_attr_views_) { item->OnNotify(); }
}

void AttributeViewBundle::Attach(AttributeView *p) {
    if (p != nullptr && m_pimpl_->m_attr_views_.emplace(p).second) {
        p->Connect(this);
        db()->Link("Attributes/" + p->name(), p->db("desc"));
        Click();
    }
}

void AttributeViewBundle::Detach(AttributeView *p) {
    if (p != nullptr && m_pimpl_->m_attr_views_.erase(p) > 0) {
        p->Disconnect();
        Click();
    }
}

bool AttributeViewBundle::isModified() {
    return SPObject::isModified() || (m_pimpl_->m_domain_ != nullptr && m_pimpl_->m_domain_->isModified());
}

bool AttributeViewBundle::Update() { return SPObject::Update(); }
DomainView const &AttributeViewBundle::GetDomain() const { return *m_pimpl_->m_domain_; }
MeshView const &AttributeViewBundle::GetMesh() const { return *m_pimpl_->m_domain_->GetMesh(); }
std::shared_ptr<DataBlock> &AttributeViewBundle::GetDataBlock(id_type guid) const {
    return m_pimpl_->m_domain_->GetDataBlock(guid);
}

void AttributeViewBundle::ForEach(std::function<void(AttributeView *)> const &fun) const {
    for (auto *attr : m_pimpl_->m_attr_views_) { fun(attr); }
}
//
// id_type AttributeDesc::GenerateGUID(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF,
//                                    int tag) {
//    std::string str = name_s + '.' + t_id.name() + '.' + static_cast<char>(IFORM + '0') + '.' +
//                      static_cast<char>(DOF + '0') + '.' + static_cast<char>(tag + '0');
//    return static_cast<id_type>(std::hash<std::string>{}(str));
//}

struct AttributeView::pimpl_s {
    AttributeViewBundle *m_bundle_;
    MeshView const *m_mesh_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    mutable std::shared_ptr<DataBlock> m_data_ = nullptr;
};
AttributeView::AttributeView() : m_pimpl_(new pimpl_s){};
// AttributeView::AttributeView(MeshView const *m) : AttributeView(){};
AttributeView::~AttributeView() { Disconnect(); }

id_type AttributeView::GetGUID() const { return GetDBValue<id_type>("GUID"); }
int AttributeView::GetTag() const { return GetDBValue<int>("Tag", 0); }

void AttributeView::Connect(AttributeViewBundle *b) {
    if (b != nullptr && b != m_pimpl_->m_bundle_) { b->Attach(this); }
    m_pimpl_->m_bundle_ = b;
}
void AttributeView::Disconnect() {
    if (m_pimpl_->m_bundle_ != nullptr) { m_pimpl_->m_bundle_->Detach(this); }
    m_pimpl_->m_bundle_ = nullptr;
}
void AttributeView::OnNotify() {
    if (m_pimpl_->m_bundle_ != nullptr) {
        m_pimpl_->m_mesh_ = &m_pimpl_->m_bundle_->GetMesh();
        m_pimpl_->m_data_ = m_pimpl_->m_bundle_->GetDataBlock(GetGUID());
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
    os << std::setw(indent + 1) << " " << name();
    return os;
};

}  //{ namespace engine
}  // namespace simpla
