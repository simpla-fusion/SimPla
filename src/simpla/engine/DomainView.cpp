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
    id_type m_current_block_id_ = NULL_ID;
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<MeshView> m_mesh_;
    std::shared_ptr<Worker> m_worker_;
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
void DomainView::Dispatch(Patch const &d) {
    ASSERT(m_pimpl_->m_mesh_ != nullptr);
    m_pimpl_->m_mesh_block_ = d.mesh_block();
};
id_type DomainView::current_block_id() const { return m_pimpl_->m_current_block_id_; }

bool DomainView::isUpdated() const {
    return m_pimpl_->m_mesh_block_ != nullptr && m_pimpl_->m_mesh_block_->id() == current_block_id() &&
           current_block_id() != NULL_ID;
}
void DomainView::Update() {
    if (isUpdated()) { return; }
    if (m_pimpl_->m_mesh_ != nullptr) m_pimpl_->m_mesh_->Update();
    if (m_pimpl_->m_worker_ != nullptr) m_pimpl_->m_worker_->Update();
    m_pimpl_->m_current_block_id_ = m_pimpl_->m_mesh_->current_block_id();
}

// void DomainView::Evaluate() {
//    if (m_pimpl_->m_worker_ != nullptr) { m_pimpl_->m_worker_->Evaluate(); }
//}

void DomainView::SetMesh(std::shared_ptr<MeshView> const &m) {
    m_pimpl_->m_mesh_ = m;
    m_pimpl_->m_mesh_->SetDomain(this);
};
std::shared_ptr<MeshView> const &DomainView::GetMesh() const { return m_pimpl_->m_mesh_; }
void DomainView::AppendWorker(std::shared_ptr<Worker> const &w) {
    if (w == nullptr) { return; }
    w->SetDomain(this);
    std::shared_ptr<Worker> &p = m_pimpl_->m_worker_;
    while (p != nullptr) { p = p->next(); }
    p = w;
};
void DomainView::PrependWorker(std::shared_ptr<Worker> const &w) {
    if (w == nullptr) { return; }
    w->SetDomain(this);
    w->next() = m_pimpl_->m_worker_;
    m_pimpl_->m_worker_ = w;
};
void DomainView::RemoveWorker(std::shared_ptr<Worker> const &w) {
    std::shared_ptr<Worker> &p = m_pimpl_->m_worker_;
    while (p != nullptr) {
        if (p != w) { p = p->next(); }
        p = p->next();
        break;
    }
};

std::shared_ptr<MeshBlock> const &DomainView::mesh_block() const { return m_pimpl_->m_mesh_block_; };
std::shared_ptr<DataBlock> DomainView::data_block(id_type) const {}
void DomainView::data_block(id_type, std::shared_ptr<DataBlock> const &) {}
std::ostream &DomainView::Print(std::ostream &os, int indent) const {
    if (m_pimpl_->m_mesh_ != nullptr) {
        os << " Mesh = { ";
        m_pimpl_->m_mesh_->Print(os, indent);
        os << " }, " << std::endl;
    }
    if (m_pimpl_->m_worker_ != nullptr) {
        os << " Worker = { ";
        m_pimpl_->m_worker_->Print(os, indent);
        os << " } " << std::endl;
    }

    return os;
};

}  // namespace engine
}  // namespace simpla