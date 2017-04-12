//
// Created by salmon on 17-2-17.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <memory>
#include "simpla/SIMPLA_config.h"
#include "simpla/data/DataTable.h"
#include "simpla/data/Serializable.h"
#include "simpla/toolbox/sp_def.h"
#include "simpla/data/EnableCreateFromDataTable.h"

namespace simpla {
namespace engine {
class Mesh;
class MeshBlock;
class Chart : public data::Serializable, public data::EnableCreateFromDataTable<Chart> {
    SP_OBJECT_BASE(Chart)
   public:
    Chart();
    virtual ~Chart();
    virtual std::shared_ptr<data::DataTable> Serialize() const;
    virtual void Deserialize(std::shared_ptr<data::DataTable> const &d);

    point_type const &GetOrigin() const;
    point_type const &GetDx() const;

    point_type map(point_type const &) const;
    point_type inv_map(point_type const &) const;
    box_type map(box_type const &) const;
    box_type inv_map(box_type const &) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_CHART_H
