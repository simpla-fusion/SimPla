/**
 * @file cube.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CUBE_H_
#define CORE_GEOMETRY_CUBE_H_

#include "simpla/SIMPLA_config.h"

#include "Body.h"
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct Cube : public Body {
    SP_GEO_OBJECT_HEAD(Cube, Body)

    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};

   protected:
    Cube() = default;
    Cube(std::initializer_list<std::initializer_list<Real>> const &v)
        : m_bound_box_(point_type(*v.begin()), point_type(*(v.begin() + 1))) {}

    template <typename V, typename U>
    Cube(V const *l, U const *h) : m_bound_box_(box_type({l[0], l[1], l[2]}, {h[0], h[1], h[2]})){};
    explicit Cube(box_type b) : m_bound_box_(std::move(b)) {}

   public:
    ~Cube() override = default;

    static std::shared_ptr<Cube> New(std::initializer_list<std::initializer_list<Real>> const &box) {
        return std::shared_ptr<Cube>(new Cube(box));
    }
    box_type GetBoundingBox() const override { return m_bound_box_; };

    bool CheckInside(point_type const &x, Real tolerance) const override {
        return std::get<0>(m_bound_box_)[0] <= x[0] && x[0] < std::get<1>(m_bound_box_)[0] &&
               std::get<0>(m_bound_box_)[1] <= x[1] && x[1] < std::get<1>(m_bound_box_)[1] &&
               std::get<0>(m_bound_box_)[2] <= x[2] && x[2] < std::get<1>(m_bound_box_)[2];
    }

    point_type Value(Real u, Real v, Real w) const override { return m_axis_.Coordinates(u, v, w); };
};

// namespace traits
//{
//
// template<typename > struct facet;
// template<typename > struct number_of_points;
//
// template<typename CS>
// struct facet<geometry::Primitive<1, CS, tags::Cube>>
//{
//	typedef geometry::Primitive<0, CS> value_type_info;
//};
//
// template<typename CS>
// struct facet<geometry::Primitive<2, CS, tags::Cube>>
//{
//	typedef geometry::Primitive<1, CS> value_type_info;
//};
//
// template<size_t N, typename CS>
// struct number_of_points<geometry::Primitive<N, CS, tags::Cube>>
//{
//	static constexpr size_t value = number_of_points<
//			typename facet<geometry::Primitive<N, CS, tags::Cube> >::value_type_info>::value
//			* 2;
//};
//
//} // namespace traits
// template<typename CS>
// typename traits::length_type<CS>::value_type_info distance(
//		geometry::Primitive<0, CS> const & p,
//		geometry::Primitive<1, CS> const & line_segment)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::value_type_info distance(
//		geometry::Primitive<0, CS> const & p,
//		geometry::Primitive<2, CS, tags::Cube> const & rect)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::value_type_info length(
//		geometry::Primitive<2, CS> const & rect)
//{
//}
// template<typename CS>
// typename traits::area_type<CS>::value_type_info area(
//		geometry::Primitive<2, CS, tags::Cube> const & rect)
//{
//}
// template<typename CS>
// typename traits::volume_type<CS>::value_type_info volume(
//		geometry::Primitive<3, CS, tags::Cube> const & poly)
//{
//}
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CUBE_H_ */
