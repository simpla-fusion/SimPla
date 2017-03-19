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
MeshView const *AttributeViewBundle::GetMesh() const { return m_pimpl_->m_domain_->GetMesh().get(); }
DomainView *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }

void AttributeViewBundle::SetPatch(std::shared_ptr<Patch> const &p) {
    for (auto *v : m_pimpl_->m_attr_views_) { v->SetData(p->GetDataBlock(v->GetGUID())); }
}
std::shared_ptr<Patch> AttributeViewBundle::GetPatch() const {
    auto res = std::make_shared<Patch>();
    for (auto *v : m_pimpl_->m_attr_views_) { res->SetDataBlock(v->GetGUID(), v->GetData()); }
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
    std::shared_ptr<data::DataBlock> m_data_ = nullptr;
    MeshView const *m_mesh_ = nullptr;
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

void AttributeView::SetData(std::shared_ptr<data::DataBlock> const &d, std::shared_ptr<MeshBlock> const &mblk) {
    m_pimpl_->m_data_ = d;
    Update();
}
std::shared_ptr<data::DataBlock> AttributeView::GetData() { return m_pimpl_->m_data_; }
std::shared_ptr<data::DataBlock> AttributeView::GetData() const { return m_pimpl_->m_data_; }
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
bool AttributeView::Update() {
    if (m_pimpl_->m_bundle_ != nullptr) { m_pimpl_->m_mesh_ = m_pimpl_->m_bundle_->GetMesh(); }
    return SPObject::Update();
}

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }

}  //{ namespace engine
}  // namespace simpla
