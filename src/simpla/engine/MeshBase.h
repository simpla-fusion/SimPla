//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/concept/Printable.h>
#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include "Attribute.h"
#include "SPObject.h"

namespace simpla {
namespace engine {
class Patch;
class Chart;
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
class MeshBase : public SPObject, public data::EnableCreateFromDataTable<MeshBase, Domain *> {
    SP_OBJECT_HEAD(MeshBase, SPObject);

   public:
    explicit MeshBase(Domain *d);
    ~MeshBase() override;
    SP_DEFAULT_CONSTRUCT(MeshBase);
    DECLARE_REGISTER_NAME("MeshBase");

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<DataTable> &t) override;

    Domain *GetDomain() const;

    virtual void InitializeData(Real time_now);
    virtual void SetBoundaryCondition(Real time_now, Real time_dt);

    id_type GetBlockId() const;

    void SetChart(std::shared_ptr<Chart> const &);
    std::shared_ptr<Chart> GetChart() const;
    void SetBlock(std::shared_ptr<MeshBlock>);
    std::shared_ptr<MeshBlock> GetBlock() const;

    void Update() override { Tag(); };

    virtual point_type point(EntityId s) const = 0;
    virtual point_type point(EntityId id, point_type const &pr) const { return point_type{}; };

    virtual point_type map(point_type const &x) const = 0;
    virtual point_type inv_map(point_type const &x) const = 0;

    virtual void RegisterRanges(std::map<std::string, EntityRange> &ranges,
                                std::shared_ptr<geometry::GeoObject> const &g = nullptr,
                                std::string const &prefix = ""){};
    virtual index_tuple GetGhostWidth(int tag = VERTEX) const;
    virtual index_box_type GetIndexBox(int tag = VERTEX) const;
    virtual box_type GetBox() const { return box_type{{0, 0, 0}, {1, 1, 1}}; }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace engine
}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
