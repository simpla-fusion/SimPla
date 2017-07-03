//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/concept/Printable.h>
#include <simpla/data/all.h>
#include <simpla/geometry/Chart.h>
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "SPObject.h"

namespace simpla {
namespace engine {
class Patch;
class MeshBase;
class MeshBlock;

using namespace simpla::data;
/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshBase : public SPObject, public AttributeGroup, public data::EnableCreateFromDataTable<MeshBase> {
    SP_OBJECT_HEAD(MeshBase, SPObject);

   public:
    explicit MeshBase(std::shared_ptr<geometry::Chart> const &c = nullptr, std::string const &s_name = "");
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    DECLARE_REGISTER_NAME(MeshBase);

    std::shared_ptr<DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &t) override;

    void Push(Patch *) override;
    void Pull(Patch *) override;

    MeshBase *GetMesh() override { return this; };
    MeshBase const *GetMesh() const override { return this; };

    virtual unsigned int GetNDims() const { return 3; }

    void SetChart(std::shared_ptr<geometry::Chart> const &);
    std::shared_ptr<geometry::Chart> GetChart() const;

    point_type const &GetCellWidth() const;
    point_type const &GetOrigin() const;

    void FitBoundBox(box_type const &);

    index_tuple GetIndexOffset() const;
    void SetDimensions(index_tuple const &);
    index_tuple GetDimensions() const;

    void SetPeriodicDimension(size_tuple const &x);
    size_tuple const &GetPeriodicDimension() const;

    void SetDefaultGhostWidth(index_tuple const &);
    index_tuple GetDefaultGhostWidth() const;
    index_tuple GetGhostWidth(int tag = VERTEX) const;

    void SetBlock(const MeshBlock &);
    const MeshBlock &GetBlock() const;
    id_type GetBlockId() const;

    box_type GetBox() const;

    virtual index_box_type GetIndexBox(int tag = VERTEX) const = 0;

    virtual point_type point(EntityId s) const = 0;
    virtual point_type local_coordinates(EntityId s, Real const *r) const = 0;
    virtual point_type global_coordinates(EntityId s, Real const *r) const;

    point_type local_coordinates(EntityId s, point_type const &r) const { return local_coordinates(s, &r[0]); };
    point_type global_coordinates(EntityId s, point_type const &r) const { return global_coordinates(s, &r[0]); };

    virtual void RegisterRanges(std::shared_ptr<geometry::GeoObject> const &g, std::string const &prefix = "");

    std::shared_ptr<std::map<std::string, EntityRange>> GetRanges();
    std::shared_ptr<std::map<std::string, EntityRange>> GetRanges() const;
    void SetRanges(std::shared_ptr<std::map<std::string, EntityRange>> const &r);

    EntityRange GetRange(std::string const &k = "") const;
    EntityRange GetBodyRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetBoundaryRange(int IFORM = VERTEX, std::string const &k = "", bool is_parallel = true) const;
    EntityRange GetParallelBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;
    EntityRange GetPerpendicularBoundaryRange(int IFORM = VERTEX, std::string const &k = "") const;

    EntityRange GetInnerRange(int IFORM = VERTEX) const;
    EntityRange GetGhostRange(int IFORM = VERTEX) const;

    void Update() override;
    virtual void InitializeData(Real time_now);
    virtual void SetBoundaryCondition(Real time_now, Real time_dt);

    index_tuple Unpack(EntityId s) const { return index_tuple{s.x, s.y, s.z}; }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
