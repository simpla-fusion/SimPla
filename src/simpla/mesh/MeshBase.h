//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_MESHBASE_H
#define SIMPLA_MESHBASE_H

#include <simpla/data/all.h>
#include <simpla/engine/Attribute.h>
#include <simpla/model/Chart.h>
#include <simpla/model/GeoObject.h>
#include <simpla/utilities/SPObject.h>

namespace simpla {

namespace engine {
class AttributeGroup;
class SPObject;
}

struct MeshBase;

/**
 *  Define:
 *   A bundle is a triple \f$(E, p, B)\f$ where \f$E\f$, \f$B\f$ are sets and \f$p:E \rightarrow B\f$ a map
 *   - \f$E\f$ is called the total space
 *   - \f$B\f$ is the base space of the bundle
 *   - \f$p\f$ is the projection
 *
 */
class MeshBase {
   public:
    SP_OBJECT_BASE(MeshBase)
    SP_DEFAULT_CONSTRUCT(MeshBase);

    explicit MeshBase(std::shared_ptr<model::Chart> const &c = nullptr, std::string const &s_name = "");

    virtual ~MeshBase();
    virtual bool empty() const { return false; }
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(const std::shared_ptr<data::DataTable> &t);
    virtual unsigned int GetNDims() const { return 3; }

    void SetChart(std::shared_ptr<model::Chart> const &);
    std::shared_ptr<model::Chart> GetChart() const;

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

    void SetBlock(const engine::MeshBlock &);
    const engine::MeshBlock &GetBlock() const;
    id_type GetBlockId() const;

    box_type GetBox() const;

    virtual void DoUpdate() {}

    virtual index_box_type GetIndexBox(int tag) const = 0;

    virtual point_type local_coordinates(EntityId s, Real const *r) const = 0;

    virtual point_type map(point_type const &) const;

    point_type local_coordinates(EntityId s, point_type const &r) const { return local_coordinates(s, &r[0]); };

    template <typename... Args>
    point_type point(Args &&... args) const {
        return local_coordinates(std::forward<Args>(args)...);
    }
    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    }

   protected:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

}  // namespace simpla

#endif  // SIMPLA_MESHBASE_H
