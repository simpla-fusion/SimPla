/**
 * @file cube.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CUBE_H_
#define CORE_GEOMETRY_CUBE_H_

#include <simpla/SIMPLA_config.h>
#include <simpla/engine/SPObject.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {

struct Cube : public GeoObject {
    SP_OBJECT_HEAD(Cube, GeoObject)
    DECLARE_REGISTER_NAME(Cube);

    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};

    Cube() : GeoObject() {}
    Cube(std::initializer_list<std::initializer_list<Real>> const &v)
        : m_bound_box_(point_type(*v.begin()), point_type(*(v.begin() + 1))) {}

    template <typename V, typename U>
    Cube(V const *l, U const *h) : m_bound_box_(box_type({l[0], l[1], l[2]}, {h[0], h[1], h[2]})){};
    Cube(box_type const &b) : m_bound_box_(b) {}

    virtual ~Cube() {}

    std::shared_ptr<data::DataTable> Pack() const override {
        auto p = std::make_shared<data::DataTable>();
        p->SetValue<std::string>("Type", GetRegisterName());
        p->SetValue("Box", m_bound_box_);
        return p;
    };
    void Unpack(std::shared_ptr<data::DataTable> const &d) override {
        if (d->has("Box")) {
            m_bound_box_ = d->GetValue<box_type>("Box");
        } else {
            std::get<0>(m_bound_box_) = d->GetValue<nTuple<Real, 3>>("lo", std::get<0>(m_bound_box_));
            std::get<1>(m_bound_box_) = d->GetValue<nTuple<Real, 3>>("hi", std::get<1>(m_bound_box_));
        }
    }

    box_type GetBoundBox() const override { return m_bound_box_; };

    virtual bool CheckInside(point_type const &x) const override {
        return std::get<0>(m_bound_box_)[0] <= x[0] && x[0] < std::get<1>(m_bound_box_)[0] &&
               std::get<0>(m_bound_box_)[1] <= x[1] && x[1] < std::get<1>(m_bound_box_)[1] &&
               std::get<0>(m_bound_box_)[2] <= x[2] && x[2] < std::get<1>(m_bound_box_)[2];
    }
};

// namespace traits
//{
//
// template<typename > struct facet;
// template<typename > struct number_of_points;
//
// template<typename CS>
// struct facet<model::Primitive<1, CS, tags::Cube>>
//{
//	typedef model::Primitive<0, CS> value_type_info;
//};
//
// template<typename CS>
// struct facet<model::Primitive<2, CS, tags::Cube>>
//{
//	typedef model::Primitive<1, CS> value_type_info;
//};
//
// template<size_t N, typename CS>
// struct number_of_points<model::Primitive<N, CS, tags::Cube>>
//{
//	static constexpr size_t value = number_of_points<
//			typename facet<model::Primitive<N, CS, tags::Cube> >::value_type_info>::value
//			* 2;
//};
//
//} // namespace traits
// template<typename CS>
// typename traits::length_type<CS>::value_type_info distance(
//		model::Primitive<0, CS> const & p,
//		model::Primitive<1, CS> const & line_segment)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::value_type_info distance(
//		model::Primitive<0, CS> const & p,
//		model::Primitive<2, CS, tags::Cube> const & rect)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::value_type_info length(
//		model::Primitive<2, CS> const & rect)
//{
//}
// template<typename CS>
// typename traits::area_type<CS>::value_type_info area(
//		model::Primitive<2, CS, tags::Cube> const & rect)
//{
//}
// template<typename CS>
// typename traits::volume_type<CS>::value_type_info volume(
//		model::Primitive<3, CS, tags::Cube> const & poly)
//{
//}
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CUBE_H_ */
