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
struct AttributeDataBase::pimpl_s {
    std::map<id_type, std::shared_ptr<AttributeDesc>> m_db_;
};
AttributeDataBase::AttributeDataBase() : m_pimpl_(new pimpl_s) {}
AttributeDataBase::~AttributeDataBase() {}
std::ostream &AttributeDataBase::Print(std::ostream &os, int indent) const {
    for (auto const &item : m_pimpl_->m_db_) {
        os << item.second->name() << "= { config= " << item.second->db() << "  },";
    }
}

bool AttributeDataBase::has(id_type id) const { return m_pimpl_->m_db_.find(id) != m_pimpl_->m_db_.end(); }
std::shared_ptr<AttributeDesc> AttributeDataBase::Get(id_type id) const { return m_pimpl_->m_db_.at(id); }

std::shared_ptr<AttributeDesc> AttributeDataBase::Set(std::shared_ptr<AttributeDesc> p) {
    auto res = m_pimpl_->m_db_.emplace(p->GUID(), p);
    if (!res.second) { res.first->second->db().merge(p->db()); }
    return res.first->second;
}
struct AttributeViewBundle::pimpl_s {
    DomainView *m_domain_ = nullptr;
    MeshView const *m_mesh_ = nullptr;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle() : m_pimpl_(new pimpl_s) {}
AttributeViewBundle::~AttributeViewBundle() {}
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { os << attr->description()->name() << " , "; }
    return os;
};
void AttributeViewBundle::insert(AttributeView *attr) {
    Click();
    m_pimpl_->m_attr_views_.insert(attr);
}
void AttributeViewBundle::erase(AttributeView *attr) {
    Click();
    m_pimpl_->m_attr_views_.erase(attr);
}
void AttributeViewBundle::insert(AttributeViewBundle *attr_bundle) {
    Click();
    m_pimpl_->m_attr_views_.insert(attr_bundle->m_pimpl_->m_attr_views_.begin(),
                                   attr_bundle->m_pimpl_->m_attr_views_.end());
}

void AttributeViewBundle::SetDomain(DomainView *d) {
    Click();
    m_pimpl_->m_domain_ = d;
}
DomainView *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }
void AttributeViewBundle::SetMesh(MeshView const *m) {
    Click();
    m_pimpl_->m_mesh_ = m;
}
MeshView const *AttributeViewBundle::GetMesh() const { return m_pimpl_->m_mesh_; }

void AttributeViewBundle::Update() {
    if (isModified()) {
        if (m_pimpl_->m_mesh_ == nullptr && m_pimpl_->m_domain_ != nullptr) { SetMesh(m_pimpl_->m_domain_->GetMesh()); }
        for (AttributeView *attr : m_pimpl_->m_attr_views_) {
            if (m_pimpl_->m_domain_ != nullptr) { attr->SetDomain(m_pimpl_->m_domain_); }
            if (m_pimpl_->m_mesh_ != nullptr) { attr->SetMesh(m_pimpl_->m_mesh_); }
        }
    }
    for (AttributeView *attr : m_pimpl_->m_attr_views_) { attr->Update(); }
    concept::StateCounter::Recount();
}
void AttributeViewBundle::RegisterAttribute(AttributeDataBase *dbase) {
    for (auto &attr : m_pimpl_->m_attr_views_) { attr->RegisterAttribute(dbase); }
}
void AttributeViewBundle::for_each(std::function<void(AttributeView *)> const &fun) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { fun(attr); }
}

id_type AttributeDesc::GenerateGUID(std::string const &name_s, std::type_info const &t_id, int IFORM, int DOF,
                                    AttributeTag tag) {
    std::string str = name_s + '.' + t_id.name() + '.' + static_cast<char>(IFORM + '0') + '.' +
                      static_cast<char>(DOF + '0') + '.' + static_cast<char>(tag + '0');
    return static_cast<id_type>(std::hash<std::string>{}(str));
}

AttributeDesc::AttributeDesc(const std::type_info &t_id, int IFORM, int DOF, AttributeTag tag,
                             const std::string &name_s)
    : m_name_(name_s),
      m_value_type_info_(t_id),
      m_iform_(IFORM),
      m_dof_(DOF),
      m_tag_(tag),
      m_GUID_(GenerateGUID(name_s, t_id, IFORM, DOF, tag)) {}

AttributeDesc::~AttributeDesc() {}

struct AttributeView::pimpl_s {
    std::shared_ptr<AttributeDesc> m_desc_;
    AttributeViewBundle *m_bundle_ = nullptr;
    DomainView *m_domain_ = nullptr;
    MeshView const *m_mesh_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<DataBlock> m_data_ = nullptr;
};

AttributeView::AttributeView(std::shared_ptr<AttributeDesc> const &desc) : m_pimpl_(new pimpl_s) {
    m_pimpl_->m_desc_ = desc;
}
AttributeView::~AttributeView() { Disconnect(); }

void AttributeView::RegisterAttribute(AttributeDataBase *dbase) { m_pimpl_->m_desc_ = dbase->Set(m_pimpl_->m_desc_); }

void AttributeView::Connect(AttributeViewBundle *b) {
    if (b != nullptr) {
        m_pimpl_->m_bundle_ = b;
        b->insert(this);
    }
}
void AttributeView::Disconnect() {
    if (m_pimpl_->m_bundle_ != nullptr) m_pimpl_->m_bundle_->erase(this);
    m_pimpl_->m_bundle_ = nullptr;
}

std::type_index AttributeView::mesh_type_index() const { return std::type_index(typeid(MeshView)); }
std::shared_ptr<AttributeDesc> const &AttributeView::description() const { return m_pimpl_->m_desc_; }
void AttributeView::SetMesh(MeshView const *p) {
    Click();
    Finalize();
    m_pimpl_->m_mesh_ = p;
};
MeshView const *AttributeView::GetMesh() const { return m_pimpl_->m_mesh_; };
void AttributeView::SetDomain(DomainView *d) {
    Click();
    Finalize();
    m_pimpl_->m_domain_ = d;
};
DomainView const *AttributeView::GetDomain() const { return m_pimpl_->m_domain_; }
DomainView *AttributeView::GetDomain() { return m_pimpl_->m_domain_; }
std::shared_ptr<DataBlock> const &AttributeView::GetDataBlock() const { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> AttributeView::GetDataBlock() { return m_pimpl_->m_data_; }

bool AttributeView::isUpdated() const {
    return isModified() || (m_pimpl_->m_data_ == nullptr) ||
           ((GetDomain() != nullptr) && (GetDomain()->GetMeshBlockId() != m_pimpl_->m_current_block_id_));
};

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
void AttributeView::Update() {
    if (m_pimpl_->m_mesh_ == nullptr && m_pimpl_->m_domain_ != nullptr) {
        m_pimpl_->m_mesh_ = m_pimpl_->m_domain_->GetMesh();
    }
    ASSERT(m_pimpl_->m_mesh_ != nullptr);

    if ((m_pimpl_->m_domain_ != nullptr) && (m_pimpl_->m_mesh_->GetMeshBlockId() != m_pimpl_->m_current_block_id_)) {
        Finalize();
        m_pimpl_->m_data_ = m_pimpl_->m_domain_->GetDataBlock(description()->GUID());
        m_pimpl_->m_current_block_id_ = m_pimpl_->m_mesh_->GetMeshBlockId();
    }
    if (m_pimpl_->m_data_ == nullptr) {
        m_pimpl_->m_data_ = CreateDataBlock();
        if (m_pimpl_->m_domain_ != nullptr) {
            m_pimpl_->m_domain_->SetDataBlock(description()->GUID(), m_pimpl_->m_data_);
        }
    }
    Initialize();
    concept::StateCounter::Recount();
}

void AttributeView::Initialize() {}
void AttributeView::Finalize() {}

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }

std::ostream &AttributeView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " " << description()->name();
    return os;
};

}  //{ namespace engine
}  // namespace simpla
