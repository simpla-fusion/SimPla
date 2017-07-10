//
// Created by salmon on 17-5-29.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <simpla/data/all.h>
#include <simpla/geometry/GeoObject.h>
#include <simpla/utilities/SPObject.h>
#include <simpla/utilities/Signal.h>
namespace simpla {
namespace geometry {
struct Chart : public engine::SPObject, public data::EnableCreateFromDataTable<Chart> {
    SP_OBJECT_HEAD(Chart, engine::SPObject)
    SP_DEFAULT_CONSTRUCT(Chart);
    DECLARE_REGISTER_NAME(Chart)

   public:
    explicit Chart(point_type shift = point_type{0, 0, 0}, point_type scale = point_type{1, 1, 1},
                   point_type rotate = point_type{0, 0, 0});
    ~Chart() override;
    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override;

    void SetShift(point_type const &x);
    point_type const &GetShift() const;

    void SetScale(point_type const &x);
    point_type const &GetScale() const;

    void SetRotation(point_type const &x);
    point_type const &GetRotation() const;

    virtual point_type GetOrigin() const;

    virtual point_type GetCellWidth(int level = 0) const;

    virtual point_type map(point_type const &) const;

    virtual point_type inv_map(point_type const &) const;

    virtual Real length(point_type const &p0, point_type const &p1) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, point_type const &p2) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1, point_type const &p2,
                        point_type const &p3) const = 0;

    virtual Real length(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1) const = 0;

    virtual Real inner_product(point_type const &uvw, vector_type const &v0, vector_type const &v1) const = 0;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};
}
}
#endif  // SIMPLA_CHART_H
