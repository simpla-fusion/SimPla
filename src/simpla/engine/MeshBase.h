//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/concept/Printable.h>
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "SPObject.h"
#include "simpla/data/EnableCreateFromDataTable.h"

namespace simpla {
namespace engine {
class MeshBlock;
class Patch;
class Chart;
class MeshBase;
/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshBase : public AttributeGroup,
                 public data::Serializable,
                 public data::EnableCreateFromDataTable<MeshBase, std::shared_ptr<Chart>> {
    SP_OBJECT_HEAD(MeshBase, AttributeGroup);

   public:
    explicit MeshBase(std::shared_ptr<Chart> c = nullptr);
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    DECLARE_REGISTER_NAME("MeshBase");

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    void Push(std::shared_ptr<Patch> p);
    std::shared_ptr<Patch> Pop();

    virtual void InitializeData(Real time_now);

    virtual void SetUp();
    virtual void TearDown();
    virtual void Initialize();
    virtual void Finalize();

    Real GetTime() const;
    id_type GetBlockId() const;
    void SetBlock(std::shared_ptr<MeshBlock>);
    std::shared_ptr<MeshBlock> GetBlock() const;

    void SetGeoObject(std::shared_ptr<geometry::GeoObject>);
    std::shared_ptr<geometry::GeoObject> GetGeoObject() const;

    void SetChart(std::shared_ptr<Chart> c);
    std::shared_ptr<Chart> GetChart() const;

    virtual Range<EntityId> GetRange(int iform) const;

    virtual Real volume(EntityId s) const = 0;
    virtual Real dual_volume(EntityId s) const = 0;
    virtual Real inv_volume(EntityId s) const = 0;
    virtual Real inv_dual_volume(EntityId s) const = 0;

    virtual point_type point(EntityId s) const = 0;
    virtual point_type point(EntityId id, point_type const &pr) const { return point_type{}; };

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
class BoundaryMeshBase : public MeshBase {
    SP_OBJECT_HEAD(BoundaryMeshBase, MeshBase);

   public:
    typedef Real scalar_type;
    explicit BoundaryMeshBase(MeshBase *m = nullptr) : m_base_mesh_(m){};
    ~BoundaryMeshBase() override = default;

    SP_DEFAULT_CONSTRUCT(BoundaryMeshBase)
    DECLARE_REGISTER_NAME("BoundaryMeshBase");

    void SetBaseMesh(MeshBase *m) { m_base_mesh_ = m; }
    const MeshBase *GetBaseMesh() const { return m_base_mesh_; }

    void SetUp() override;

    Range<EntityId> &GetRange(int IFORM) { return m_range_[IFORM]; };
    Range<EntityId> GetRange(int IFORM) const override { return m_range_[IFORM]; };

    point_type point(EntityId s) const override { return GetBaseMesh()->point(s); }
    Real volume(EntityId s) const override { return m_base_mesh_->volume(s); }
    Real dual_volume(EntityId s) const override { return m_base_mesh_->volume(s); }
    Real inv_volume(EntityId s) const override { return m_base_mesh_->volume(s); }
    Real inv_dual_volume(EntityId s) const override { return m_base_mesh_->volume(s); }

   private:
    MeshBase *m_base_mesh_ = nullptr;
    Range<EntityId> m_range_[4];
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
