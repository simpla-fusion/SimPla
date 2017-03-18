/**
 * @file cube.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_CUBE_H_
#define CORE_GEOMETRY_CUBE_H_

namespace simpla {
namespace geometry {

struct Cube : public GeoObject {
    Cube(std::initializer_list<std::initializer_list<Real>> const &v)
    //        : GeoObject::m_bound_box_(point_type(*v.begin()), point_type(*(v.begin() + 1)))
    {}
    Cube(box_type const &b)
    //            : m_bound_box_(b)
    {}
    ~Cube() {}
    point_type lower() const { return std::get<0>(GeoObject::GetBoundBox()); }
    point_type upper() const { return std::get<1>(GeoObject::GetBoundBox()); }

    virtual Real distance(point_type const &x) const { return 0; }
    /**
     * @brief >0 out ,=0 surface ,<0 in
     * @param x
     * @return
     */
    virtual int isInside(point_type const &x) const { return 0; }
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
