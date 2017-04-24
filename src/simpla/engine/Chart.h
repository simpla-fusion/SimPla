//
// Created by salmon on 17-2-17.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <simpla/utilities/sp_def.h>
#include <memory>
#include "simpla/data/DataTable.h"
#include "simpla/data/EnableCreateFromDataTable.h"
#include "simpla/data/Serializable.h"

namespace simpla {
namespace engine {
class MeshBase;
class MeshBlock;
class Chart : public data::Serializable, public data::EnableCreateFromDataTable<Chart> {
    SP_OBJECT_BASE(engine::Chart)
   public:
    Chart();
    ~Chart() override;
    SP_DEFAULT_CONSTRUCT(Chart)
    DECLARE_REGISTER_NAME("Chart")

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> t) override;

    void SetOrigin(point_type const &);
    void SetDx(point_type const &);
    point_type const &GetOrigin() const;
    point_type const &GetDx() const;

    point_type map(point_type const &) const;
    point_type inv_map(point_type const &) const;
    point_type inv_map(index_tuple const &) const;
    box_type map(box_type const &) const;
    box_type inv_map(box_type const &) const;
    box_type inv_map(index_box_type const &) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}  // namespace engine{
}  // namespace simpla{
#endif  // SIMPLA_CHART_H
