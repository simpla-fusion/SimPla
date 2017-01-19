//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/design_pattern/Observer.h>
#include "Attribute.h"
#include "MeshBlock.h"

namespace simpla {

namespace model {
class Model;
}  // namespace model{

namespace mesh {
class Patch;
class DataBlock;

/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class Mesh : public concept::Printable, public concept::LifeControllable {
   public:
    SP_OBJECT_BASE(Mesh);
    typedef MeshEntityId entity_id;

    Mesh();

    virtual ~Mesh();

    virtual std::ostream& Print(std::ostream &os, int indent) const;

    virtual void Deploy();

    virtual void PreProcess();

    virtual void PostProcess();

    virtual void Initialize(Real data_time = 0, Real dt = 0);

    virtual void Finalize(Real data_time = 0, Real dt = 0);

    virtual std::shared_ptr<MeshBlock> const& mesh_block() const {
        ASSERT(m_mesh_block_ != nullptr);
        return m_mesh_block_;
    }

    decltype(auto) dimensions() const { return m_mesh_block_->dimensions(); }

    template <typename... Args>
    Range<MeshEntityId> range(Args&&... args) const {
        if (m_mesh_block_ != nullptr) {
            return m_mesh_block_->range(std::forward<Args>(args)...);
        } else {
            return Range<MeshEntityId>();
        }
    }
    size_type size(int IFORM = VERTEX) const { return m_mesh_block_->number_of_entities(IFORM); }
    template <typename... Args>
    auto hash(Args&&... args) const {
        return m_mesh_block_->hash(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto pack(Args&&... args) const {
        return m_mesh_block_->pack(std::forward<Args>(args)...);
    }
    point_type dx() const {
        if (m_mesh_block_ != nullptr) {
            return m_mesh_block_->dx();
        } else {
            return point_type{1, 1, 1};
        }
    }
    template <typename... Args>
    decltype(auto) point_global_to_local(Args&&... args) const {
        return m_mesh_block_->point_global_to_local(std::forward<Args>(args)...);
    }
    template <typename... Args>
    decltype(auto) point(Args&&... args) const {
        return m_mesh_block_->point(std::forward<Args>(args)...);
    }

    simpla::model::Model& model() {
        ASSERT(m_model_ != nullptr);
        return *m_model_;
    }
    simpla::model::Model const& model() const {
        ASSERT(m_model_ != nullptr);
        return *m_model_;
    }

    virtual void Connect(Attribute *attr) { m_attrs_.insert(attr); };
    virtual void Disconnect(Attribute *attr) { m_attrs_.erase(attr); }

    /**
     *
     * @brief
     * @param p
     *
     * @startuml{ZipCmd_ZipComp_Communication.png}
     *
     * ZipCmd -> ZipComp: First Compute Request
     * ZipCmd <-- ZipComp: First Compute Response
     *
     * ZipCmd -> ZipComp: Second Compute Request
     * ZipCmd <-- ZipComp: Second Compute Response
     *
     * @enduml

     */
    virtual void Accept(Patch *p);
    template <typename TFun>
    void ForeachAttr(TFun const &fun) {
        for (auto attr : m_attrs_) { fun(attr); }
    }

   protected:
    std::unique_ptr<simpla::model::Model> m_model_;
    std::set<Attribute*> m_attrs_;
    std::shared_ptr<MeshBlock> m_mesh_block_;
};

}  // namespace mesh
}  // namespace simpla

#endif  // SIMPLA_GEOMETRY_H
