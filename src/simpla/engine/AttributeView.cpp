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
    std::map<std::string, id_type> m_name_map_;
};
AttributeDataBase::AttributeDataBase() : m_pimpl_(new pimpl_s) {}
AttributeDataBase::~AttributeDataBase() {}
std::ostream &AttributeDataBase::Print(std::ostream &os, int indent) const {
    for (auto const &item : m_pimpl_->m_db_) {
        os << std::setw(indent + 1) << " " << item.second->name() << " = {"
           << " iform = " << item.second->iform() << ", dof = " << item.second->dof() << ", value type = \""
           << item.second->value_type_info().name() << "\" , config= " << item.second->db() << "  }," << std::endl;
    }
}
id_type AttributeDataBase::GetGUID(std::string const &s) const { return m_pimpl_->m_name_map_.at(s); }
bool AttributeDataBase::has(id_type id) const { return m_pimpl_->m_db_.find(id) != m_pimpl_->m_db_.end(); }
bool AttributeDataBase::has(std::string const &s) const {
    return m_pimpl_->m_name_map_.find(s) != m_pimpl_->m_name_map_.end();
}
std::shared_ptr<AttributeDesc> AttributeDataBase::Get(id_type id) const { return m_pimpl_->m_db_.at(id); }
std::shared_ptr<AttributeDesc> AttributeDataBase::Get(std::string const &s) const { return Get(GetGUID(s)); }
std::shared_ptr<AttributeDesc> AttributeDataBase::Set(id_type gid, std::shared_ptr<AttributeDesc> p) {
    auto res = m_pimpl_->m_db_.emplace(gid, p);
    if (res.second) {
        m_pimpl_->m_name_map_.emplace(p->name(), gid);
    } else {
        res.first->second->db().merge(p->db());
    }
    return res.first->second;
}

void AttributeDataBase::Remove(id_type id) {
    m_pimpl_->m_name_map_.erase(Get(id)->name());
    m_pimpl_->m_db_.erase(id);
}
void AttributeDataBase::Remove(const std::string &s) {
    m_pimpl_->m_db_.erase(GetGUID(s));
    m_pimpl_->m_name_map_.erase(s);
}

void AttributeDataBase::for_each(std::function<void(AttributeDesc *)> const &fun) {
    for (auto &item : m_pimpl_->m_db_) { fun(item.second.get()); }
}
void AttributeDataBase::for_each(std::function<void(AttributeDesc const *)> const &fun) const {
    for (auto const &item : m_pimpl_->m_db_) { fun(item.second.get()); }
}
struct AttributeViewBundle::pimpl_s {
    DomainView *m_domain_ = nullptr;
    MeshView const *m_mesh_ = nullptr;
    std::set<AttributeView *> m_attr_views_;
};

AttributeViewBundle::AttributeViewBundle() : m_pimpl_(new pimpl_s) {}
AttributeViewBundle::~AttributeViewBundle() {
    for (auto &attr : m_pimpl_->m_attr_views_) { attr->Disconnect(); }
}
std::ostream &AttributeViewBundle::Print(std::ostream &os, int indent) const {
    for (auto &attr : m_pimpl_->m_attr_views_) { os << attr->description().name() << " , "; }
    return os;
};
void AttributeViewBundle::Connect(AttributeView *attr) {
    if (m_pimpl_->m_attr_views_.insert(attr).second) {
        Click();
        attr->Connect(this);
    };
}

void AttributeViewBundle::Disconnect(AttributeView *attr) {
    if (m_pimpl_->m_attr_views_.erase(attr) > 0) {
        Click();
        attr->Disconnect();
    };
}
void AttributeViewBundle::Merge(AttributeViewBundle *attr_bundle) {
    Click();
    m_pimpl_->m_attr_views_.insert(attr_bundle->m_pimpl_->m_attr_views_.begin(),
                                   attr_bundle->m_pimpl_->m_attr_views_.end());
}

void AttributeViewBundle::SetDomain(DomainView *d) {
    Click();
    m_pimpl_->m_domain_ = d;
}
DomainView *AttributeViewBundle::GetDomain() { return m_pimpl_->m_domain_; }
DomainView const *AttributeViewBundle::GetDomain() const { return m_pimpl_->m_domain_; }
void AttributeViewBundle::SetMesh(MeshView const *m) {
    Click();
    m_pimpl_->m_mesh_ = m;
}
MeshView const *AttributeViewBundle::GetMesh() const {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    return m_pimpl_->m_mesh_;
}

void AttributeViewBundle::Update() {
    if (!isModified()) { return; }
    if (m_pimpl_->m_mesh_ == nullptr && m_pimpl_->m_domain_ != nullptr) { SetMesh(m_pimpl_->m_domain_->GetMesh()); }
    for (AttributeView *attr : m_pimpl_->m_attr_views_) { attr->Update(); }
    concept::StateCounter::Recount();
}
void AttributeViewBundle::RegisterAttribute(AttributeDataBase *dbase) {
    for (auto &attr : m_pimpl_->m_attr_views_) { attr->RegisterAttribute(dbase); }
}
void AttributeViewBundle::for_each(std::function<void(AttributeView *)> const &fun) {
    for (auto *attr : m_pimpl_->m_attr_views_) { fun(attr); }
}
void AttributeViewBundle::for_each(std::function<void(AttributeView const *)> const &fun) const {
    for (auto const *attr : m_pimpl_->m_attr_views_) { fun(attr); }
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
    AttributeViewBundle *m_bundle_ = nullptr;
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<DataBlock> m_data_ = nullptr;
};
AttributeView::AttributeView() : m_pimpl_(new pimpl_s) {}
AttributeView::~AttributeView() { Disconnect(); }
void AttributeView::Config(AttributeViewBundle *b, std::string const &s, AttributeTag t) {
    Connect(b);
    m_pimpl_->m_desc_ = std::make_shared<AttributeDesc>(s, value_type_info(), iform(), dof(), t);
}
AttributeDesc const &AttributeView::description() const { return *m_pimpl_->m_desc_; }
data::DataTable &AttributeView::db() { return m_pimpl_->m_desc_->db(); }
const data::DataTable &AttributeView::db() const { return m_pimpl_->m_desc_->db(); }

void AttributeView::RegisterAttribute(AttributeDataBase *dbase) {
    m_pimpl_->m_desc_ = dbase->Set(m_pimpl_->m_desc_->GUID(), m_pimpl_->m_desc_);
}

void AttributeView::Connect(AttributeViewBundle *b) {
    if (m_pimpl_->m_bundle_ != b) {
        Disconnect();
        m_pimpl_->m_bundle_ = b;
        if (b != nullptr) { b->Connect(this); }
    }
}
void AttributeView::Disconnect() {
    if (m_pimpl_->m_bundle_ != nullptr) { m_pimpl_->m_bundle_->Disconnect(this); }
    m_pimpl_->m_bundle_ = nullptr;
}

MeshView const *AttributeView::GetMesh() const {
    ASSERT(m_pimpl_->m_bundle_ != nullptr);
    return m_pimpl_->m_bundle_->GetMesh();
};

std::shared_ptr<DataBlock> const &AttributeView::GetDataBlock() const { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> &AttributeView::GetDataBlock() { return m_pimpl_->m_data_; }
std::shared_ptr<DataBlock> AttributeView::CreateDataBlock() const {
    UNIMPLEMENTED;
    return nullptr;
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
    ASSERT(m_pimpl_->m_bundle_ != nullptr);
    if ((GetMesh()->GetMeshBlockId() != m_pimpl_->m_current_block_id_)) {
        m_pimpl_->m_current_block_id_ = GetMesh()->GetMeshBlockId();
        if (m_pimpl_->m_bundle_->GetDomain() == nullptr) {
            m_pimpl_->m_data_ = CreateDataBlock();
        } else {
            m_pimpl_->m_data_ = m_pimpl_->m_bundle_->GetDomain()->GetDataBlock(GUID());
            if (m_pimpl_->m_data_ == nullptr) {
                m_pimpl_->m_data_ = CreateDataBlock();
                m_pimpl_->m_bundle_->GetDomain()->SetDataBlock(GUID(), m_pimpl_->m_data_);
            }
        }
    }
}

bool AttributeView::isNull() const { return m_pimpl_->m_data_ == nullptr; }

id_type AttributeView::GUID() const { return m_pimpl_->m_desc_->GUID(); }
std::string const &AttributeView::name() const { return m_pimpl_->m_desc_->name(); }
std::type_info const &AttributeView::value_type_info() const { return typeid(Real); }
int AttributeView::iform() const { return VERTEX; }
int AttributeView::dof() const { return 1; }
AttributeTag AttributeView::tag() const { return m_pimpl_->m_desc_->tag(); }

std::ostream &AttributeView::Print(std::ostream &os, int indent) const {
    os << std::setw(indent + 1) << " " << description().name();
    return os;
};

}  //{ namespace engine
}  // namespace simpla
