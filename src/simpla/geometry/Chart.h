//
// Created by salmon on 17-5-29.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/Signal.h>
#include "simpla/engine/SPObject.h"
namespace simpla {
namespace geometry {
struct Chart : public SPObject, public data::EnableCreateFromDataTable<Chart, std::string const &> {
    SP_OBJECT_HEAD(Chart, SPObject)
    SP_DEFAULT_CONSTRUCT(Chart);
    DECLARE_REGISTER_NAME("Chart")

   public:
    explicit Chart(std::string const &s_name = "");
    ~Chart() override;
    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    void SetPeriodicDimension(point_type const &d);
    point_type const &GetPeriodicDimension() const;
    void SetOrigin(point_type const &x);
    void SetScale(point_type const &x);

    point_type GetOrigin() const;
    point_type GetScale() const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_CHART_H
