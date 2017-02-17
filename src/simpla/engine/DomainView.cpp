//
// Created by salmon on 17-2-12.
//
#include "DomainView.h"
#include <simpla/SIMPLA_config.h>
#include <set>
#include "AttributeView.h"
#include "MeshView.h"
#include "Object.h"
#include "Patch.h"
#include "Worker.h"

namespace simpla {
namespace engine {
struct DomainView::pimpl_s {
    int m_state_count_ = 0;
    int m_current_state_count_ = 0;
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<MeshView> m_mesh_;
    std::list<std::shared_ptr<Worker>> m_workers_;
    std::shared_ptr<Patch> m_patch_;
    //    std::map<id_type, std::shared_ptr<AttributeDesc>> m_attrs_dict_;
};

DomainView::DomainView() : m_pimpl_(new pimpl_s) {}

DomainView::~DomainView() {}

/**
 *
 * @startuml
 * actor Main
 * Main -> DomainView : Set U as MeshView
 * activate DomainView
 *     alt if MeshView=nullptr
 *          create MeshView
 *     DomainView -> MeshView : create U as MeshView
 *     MeshView --> DomainView: return MeshView
 *     end
 *     DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * actor Main
 * Main -> DomainView : Dispatch
 * activate DomainView
 *     DomainView->MeshView:  Dispatch
 *     MeshView->MeshView: SetMeshBlock
 *     activate MeshView
 *     deactivate MeshView
 *     MeshView -->DomainView:  Done
*      DomainView --> Main : Done
 * deactivate DomainView
 * @enduml
 * @startuml
 * Main ->DomainView: Update
 * activate DomainView
 *     DomainView -> AttributeView : Update
 *     activate AttributeView
 *          AttributeView -> Field : Update
 *          Field -> AttributeView : Update
 *          activate AttributeView
 *               AttributeView -> DomainView : get DataBlock at attr.id()
 *               DomainView --> AttributeView : return DataBlock at attr.id()
 *               AttributeView --> Field : return DataBlock is ready
 *          deactivate AttributeView
 *          alt if data_block.isNull()
 *              Field -> Field :  create DataBlock
 *              Field -> AttributeView : send DataBlock
 *              AttributeView --> Field : Done
 *          end
 *          Field --> AttributeView : Done
 *          AttributeView --> DomainView : Done
 *     deactivate AttributeView
 *     DomainView -> MeshView : Update
 *     activate MeshView
 *          alt if isFirstTime
 *              MeshView -> AttributeView : Set Initialize Value
 *              activate AttributeView
 *                   AttributeView --> MeshView : Done
 *              deactivate AttributeView
 *          end
 *          MeshView --> DomainView : Done
 *     deactivate MeshView
 *     DomainView -> Worker : Update
 *     activate Worker
 *          alt if isFirstTime
 *              Worker -> AttributeView : set initialize value
 *              activate AttributeView
 *                  AttributeView --> Worker : Done
 *              deactivate AttributeView
 *          end
 *          Worker --> DomainView : Done
 *     deactivate Worker
 *     DomainView --> Main : Done
 * deactivate DomainView
 * deactivate Main
 * @enduml
 */
void DomainView::Dispatch(std::shared_ptr<Patch> p) { m_pimpl_->m_patch_ = p; };

id_type DomainView::current_block_id() const { return m_pimpl_->m_current_block_id_; }

bool DomainView::isUpdated() const {
    return (m_pimpl_->m_current_block_id_ ==
            ((m_pimpl_->m_patch_ == nullptr) ? NULL_ID : m_pimpl_->m_patch_->mesh_block()->id())) &&
           (m_pimpl_->m_current_state_count_ == m_pimpl_->m_state_count_);
}

void DomainView::Update() {
    if (m_pimpl_->m_current_state_count_ < m_pimpl_->m_state_count_)
    {        //FIXME: update attr_desc
        m_pimpl_->m_current_state_count_ = m_pimpl_->m_state_count_;
    }
    if (m_pimpl_->m_patch_ == nullptr) { m_pimpl_->m_patch_ = std::make_shared<Patch>(); }
    if (m_pimpl_->m_mesh_ != nullptr) { m_pimpl_->m_mesh_->Update(); }
    for (auto &item : m_pimpl_->m_workers_) { item->Update(); }
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_patch_->mesh_block()->id();
}

void DomainView::Evaluate() {
    for (auto &item : m_pimpl_->m_workers_) { item->Evaluate(); }
}

void DomainView::SetMesh(std::shared_ptr<MeshView> const &m) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_mesh_->SetDomain(this);
    ++m_pimpl_->m_state_count_;
};

std::shared_ptr<MeshView> const &DomainView::GetMesh() const { return m_pimpl_->m_mesh_; }

void DomainView::AppendWorker(std::shared_ptr<Worker> const &w) {
    if (w == nullptr) { return; }
    w->SetDomain(this);
    m_pimpl_->m_workers_.push_back(w);
    ++m_pimpl_->m_state_count_;
};

void DomainView::PrependWorker(std::shared_ptr<Worker> const &w) {
    if (w == nullptr) { return; }
    w->SetDomain(this);
    m_pimpl_->m_workers_.push_front(w);
    ++m_pimpl_->m_state_count_;
};

void DomainView::RemoveWorker(std::shared_ptr<Worker> const &w) {
    UNIMPLEMENTED;
    ++m_pimpl_->m_state_count_;
    //    auto it = m_pimpl_->m_workers_.find(w);
    //    if (it != m_pimpl_->m_workers_.end()) { m_pimpl_->m_workers_.erase(it); }
};

std::shared_ptr<MeshBlock> const &DomainView::mesh_block() const { return m_pimpl_->m_patch_->mesh_block(); };

std::shared_ptr<DataBlock> DomainView::data_block(id_type) const {}

void DomainView::data_block(id_type, std::shared_ptr<DataBlock> const &) {}

std::ostream &DomainView::Print(std::ostream &os, int indent) const {
    if (m_pimpl_->m_mesh_ != nullptr)
    {
        os << " Mesh = { ";
        m_pimpl_->m_mesh_->Print(os, indent);
        os << " }, " << std::endl;
    }

    if (m_pimpl_->m_workers_.size() > 0)
    {
        os << " Worker = { ";
        for (auto &item : m_pimpl_->m_workers_) { item->Print(os, indent); }
        os << " } " << std::endl;
    }

    return os;
};

}  // namespace engine
}  // namespace simpla