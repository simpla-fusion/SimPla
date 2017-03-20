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
    MeshView const *m_mesh_;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle(DomainView *d) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_domain_ = d;
    if (m_pimpl_->m_domain_ != nullptr) { m_pimpl_->m_domain_->Attach(this); }
}
AttributeViewBundle::~AttributeViewBundle() {
    if (m_pimpl_->m_domain_ != nullptr) {
        m_pimpl_->m_domain_->Detach(this);
        m_pimpl_->m_domain_ = nullptr;
    }
}
void AttributeViewBundle::Attach(AttributeView *p) {
    if (p != nullptr) { m_pimpl_->m_attr_views_.emplace(p); }
}

void AttributeViewBundle::Detach(AttributeView *p) {
    if (p != nullptr) { m_pimpl_->m_attr_views_.erase(p); }
}
DomainView *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }

void AttributeViewBundle::SetMesh(MeshView const *m) {
    m_pimpl_->m_mesh_ = m;
    for (auto *v : m_pimpl_->m_attr_views_) { v->SetMesh(m); }
}
MeshView const *AttributeViewBundle::GetMesh() const { return m_pimpl_->m_mesh_; }

void AttributeViewBundle::PushData(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<data::DataEntity> const &p) {
    if (GetMesh() == nullptr || GetMesh()->GetMeshBlockId() != m->GetGUID()) {
        RUNTIME_ERROR << " data and mesh mismatch!" << std::endl;
    }

    ASSERT(p->isTable());
    auto const &t = p->cast_as<data::DataTable>();
    for (auto *v : m_pimpl_->m_attr_views_) { v->PushData(m, t.Get(std::to_string(v->GetGUID()))); }
}
std::pair<std::shared_ptr<MeshBlock>, std::shared_ptr<data::DataEntity>> AttributeViewBundle::PopData() {
    auto res = std::make_shared<data::DataTable>();
    for (auto *v : m_pimpl_->m_attr_views_) { res->Set(std::to_string(v->GetGUID()), v->PopData().second); }
    return std::make_pair(m_pimpl_->m_mesh_->GetMeshBlock(), res);
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
    MeshView const *m_mesh_;
};
AttributeView::AttributeView(AttributeViewBundle *b, std::shared_ptr<data::DataEntity> const &t)
    : SPObject(t), m_pimpl_(new pimpl_s) {
    if (b != nullptr && b != m_pimpl_->m_bundle_) { b->Attach(this); }
    m_pimpl_->m_bundle_ = b;
};

AttributeView::~AttributeView() {
    if (m_pimpl_->m_bundle_ != nullptr) { m_pimpl_->m_bundle_->Detach(this); }
    m_pimpl_->m_bundle_ = nullptr;
}

void AttributeView::Config() {
    db()->SetValue("iform", GetIFORM());
    db()->SetValue("dof", GetDOF());
    db()->SetValue("value value_type_info", value_type_info().name());
    db()->SetValue("value value_type_info idx", std::type_index(value_type_info()).hash_code());
    db()->SetValue("GUID", GenerateGUID(name(), value_type_info(), GetIFORM(), GetDOF(), GetTag()));
}

id_type AttributeView::GetGUID() const { return GetDBValue<id_type>("GUID", NULL_ID); }
int AttributeView::GetTag() const { return GetDBValue<int>("Tag", 0); }
void AttributeView::SetMesh(MeshView const *m) {
    if (m_pimpl_->m_mesh_ == nullptr || m_pimpl_->m_mesh_->GetTypeInfo() == m->GetTypeInfo()) {
        m_pimpl_->m_mesh_ = m;
    } else {
        RUNTIME_ERROR << "Can not change the mesh type of a worker! [ from " << m_pimpl_->m_mesh_->GetClassName()
                      << " to " << m->GetClassName() << "]" << std::endl;
    }
}
MeshView const *AttributeView::GetMesh() const { return m_pimpl_->m_mesh_; }
/**
*@startuml
*start
* if (m_domain_ == nullptr) then (yes)
* else   (no)
*   : m_current_block_id = m_domain-> current_block_id();
* endif
*stop
*@enduml
*/
bool AttributeView::Update() { return SPObject::Update(); }

bool AttributeView::isNull() const { return false; }

}  //{ namespace engine
}  // namespace simpla
