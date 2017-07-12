//
// Created by salmon on 17-5-29.
//

#ifndef SIMPLA_CHART_H
#define SIMPLA_CHART_H

#include <simpla/data/all.h>
#include <simpla/engine/SPObject.h>
#include <simpla/geometry/GeoObject.h>
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

    point_type GetOrigin() const;

    point_type GetCellWidth(int level = 0) const;

    template <typename TR>
    point_type local_coordinates(TR const &x) const {
        return point_type{std::fma(x[0], m_scale_[0], m_shift_[0]), std::fma(x[1], m_scale_[1], m_shift_[1]),
                          std::fma(x[2], m_scale_[2], m_shift_[2])};
    }

    template <typename TR>
    point_type local_coordinates(index_tuple x, TR const &r) const {
        return local_coordinates(point_type{x[0] + r[0], x[1] + r[1], x[2] + r[2]});
    }

    template <typename TR>
    point_type local_coordinates(index_type x, index_type y, index_type z, TR const &r) const {
        return local_coordinates(point_type{x + r[0], y + r[1], z + r[2]});
    }

    template <typename TR>
    std::tuple<index_tuple, point_type> invert_local_coordinates(TR const &x) const {
        point_type r = (x - m_shift_) / m_scale_;
        index_tuple idx = r + 0.5;
        r -= idx;
        return std::make_tuple(idx, r);
    }

    template <typename TR>
    std::tuple<index_tuple, point_type> invert_global_coordinates(TR const &x) const {
        return invert_local_coordinates(inv_map(x));
    }

    template <typename... Args>
    point_type global_coordinates(Args &&... args) const {
        return map(local_coordinates(std::forward<Args>(args)...));
    };

    virtual point_type map(point_type const &x) const { return x; }

    virtual point_type inv_map(point_type const &x) const { return x; }

    virtual Real length(point_type const &p0, point_type const &p1) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, point_type const &p2) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1, point_type const &p2,
                        point_type const &p3) const = 0;

    virtual Real length(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real area(point_type const &p0, point_type const &p1, int normal) const = 0;

    virtual Real volume(point_type const &p0, point_type const &p1) const = 0;

    virtual Real inner_product(point_type const &uvw, vector_type const &v0, vector_type const &v1) const = 0;

   private:
    point_type m_shift_{0, 0, 0};
    point_type m_rotation_{0, 0, 0};
    point_type m_scale_{1, 1, 1};
};
}
}
#endif  // SIMPLA_CHART_H
