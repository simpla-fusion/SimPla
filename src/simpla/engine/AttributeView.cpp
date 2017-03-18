//
// Created by salmon on 16-10-20.
//

#include "AttributeView.h"
#include <simpla/data/DataBlock.h>
#include <set>
#include <typeindex>
#include "DomainView.h"
#include "MeshBlock.h"
#include "MeshView.h"
#include "Patch.h"
namespace simpla {
namespace engine {

struct AttributeViewBundle::pimpl_s {
    DomainView *m_domain_ = nullptr;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle(std::shared_ptr<data::DataEntity> const &t)
    : SPObject(t), m_pimpl_(new pimpl_s) {}
AttributeViewBundle::~AttributeViewBundle() {}
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { os << attr->name() << " , "; }
    return os;
};

void AttributeViewBundle::Attach(AttributeView *p) {
    if (p != nullptr && m_pimpl_->m_attr_views_.emplace(p).second) {
        p->Connect(this);
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
DomainView *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }
void AttributeViewBundle::RegisterDomain(DomainView *d) { m_pimpl_->m_domain_ = d; }
std::shared_ptr<MeshView> AttributeViewBundle::GetMesh() const {
    ASSERT(m_pimpl_->m_domain_ != nullptr);
    return m_pimpl_->m_domain_->GetMesh();
}
std::shared_ptr<data::DataBlock> AttributeViewBundle::GetDataBlock(id_type guid) const {
    return m_pimpl_->m_domain_->GetDataBlock(guid);
}
void AttributeViewBundle::PushPatch(std::shared_ptr<Patch> const &p) {
    for (auto *v : m_pimpl_->m_attr_views_) { v->PushDataBlock(p->GetDataBlock(v->GetGUID())); }
}
std::shared_ptr<Patch> AttributeViewBundle::PopPatch() const {
    auto res = std::make_shared<Patch>();
    for (auto *v : m_pimpl_->m_attr_views_) { res->SetDataBlock(v->GetGUID(), v->PopDataBlock()); }
    return res;
}
void AttributeViewBundle::Foreach(std::function<void(AttributeView *)> const &fun) const {
    for (auto *attr : m_pimpl_->m_attr_views_) { fun(attr); }
}

id_type GenerateGUID(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF, int tag) {
    std::string str = name_s + '.' + t_id.name() + '.' + static_cast<char>(IFORM + '0') + '.' +
                      static_cast<char>(DOF + '0') + '.' + static_cast<char>(tag + '0');
    return static_cast<id_type>(std::hash<std::string>{}(str));
}

struct AttributeView::pimpl_s {
    AttributeViewBundle *m_bundle_;
    std::shared_ptr<MeshView> m_mesh_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<data::DataBlock> m_data_ = nullptr;
};
AttributeView::AttributeView(AttributeViewBundle *b, std::shared_ptr<data::DataEntity> const &t)
    : SPObject(t), m_pimpl_(new pimpl_s) {
    Connect(b);
};
AttributeView::AttributeView(std::shared_ptr<MeshView> const &m, std::shared_ptr<data::DataEntity> const &t)
    : SPObject(t), m_pimpl_(new pimpl_s) {
    m_pimpl_->m_mesh_ = m;
}
void AttributeView::Config() {
    db()->SetValue("iform", GetIFORM());
    db()->SetValue("dof", GetDOF());
    db()->SetValue("value value_type_info", GetValueTypeInfo().name());
    db()->SetValue("value value_type_info idx", std::type_index(GetValueTypeInfo()).hash_code());
    db()->SetValue("GUID", GenerateGUID(name(), GetValueTypeInfo(), GetIFORM(), GetDOF(), GetTag()));
}
// AttributeView::AttributeView(MeshView const *m) : AttributeView(){};
AttributeView::~AttributeView() { Disconnect(); }

id_type AttributeView::GetGUID() const { return GetDBValue<id_type>("GUID", NULL_ID); }
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
        m_pimpl_->m_mesh_ = m_pimpl_->m_bundle_->GetMesh();
        m_pimpl_->m_data_ = m_pimpl_->m_bundle_->GetDataBlock(m_pimpl_->m_mesh_->GetMeshBlock()->GetGUID());
    } else {
        DO_NOTHING;
    }
}

MeshView const &AttributeView::GetMesh() const {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    return *m_pimpl_->m_mesh_;
};
void AttributeView::PushDataBlock(std::shared_ptr<data::DataBlock> const &d) {
    m_pimpl_->m_data_ = d;
    if (m_pimpl_->m_data_ == nullptr) { m_pimpl_->m_data_ = std::make_shared<data::DataBlock>(); };
}
std::shared_ptr<data::DataBlock> AttributeView::PopDataBlock() { return m_pimpl_->m_data_; }

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
