/**
 * @file polygon.h
 * @author salmon
 * @date 2015-11-17.
 */

#ifndef SIMPLA_POLYGON_H
#define SIMPLA_POLYGON_H

#include <vector>

#include <simpla/data/all.h>
#include <simpla/utilities/FancyStream.h>
#include <simpla/utilities/nTuple.h>
#include <simpla/utilities/SPObject.h>
namespace simpla {
namespace geometry {
/**
 * @ingroup geometry
 * @{
 */

template <int NDIMS>
class Polygon;

/**
 *  @brief 2D polygon
 */
template <>
struct Polygon<2> : public data::Serializable {
    typedef nTuple<Real, 2> point2d_type;

    SP_OBJECT_BASE(Polygon<2>);

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

   public:
    Polygon() {}

    ~Polygon() {}

    Polygon(Polygon const &) = delete;

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = data::Serializable::Serialize();
        res->SetValue("Type", "Polygon2D");

        auto v_array = std::make_shared<data::DataEntityWrapper<point2d_type *>>();

        for (size_type s = 0, se = m_polygon_.size(); s < se; ++s) { v_array->Add(m_polygon_[s]); }

        res->Set("data", std::dynamic_pointer_cast<data::DataEntity>(v_array));
        return res;
    };
    void Deserialize(const std::shared_ptr<data::DataTable> &t) override {}

    std::vector<point2d_type> &data() { return m_polygon_; };

    std::vector<point2d_type> const &data() const { return m_polygon_; };

    void push_back(point2d_type const &p);

    void deploy();

    Real nearest_point(Real *x, Real *y) const;
    bool check_inside(Real x, Real y) const;

    std::tuple<point2d_type, point2d_type> GetBoundBox() const { return std::move(std::make_tuple(m_min_, m_max_)); };

   private:
    point2d_type m_min_, m_max_;
};
/* @} */
}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_POLYGON_H
