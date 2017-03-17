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
    point_type lower() const { return std::get<0>(GeoObject::bound_box()); }
    point_type upper() const { return std::get<1>(GeoObject::bound_box()); }

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
//	typedef model::Primitive<0, CS> type;
//};
//
// template<typename CS>
// struct facet<model::Primitive<2, CS, tags::Cube>>
//{
//	typedef model::Primitive<1, CS> type;
//};
//
// template<size_t N, typename CS>
// struct number_of_points<model::Primitive<N, CS, tags::Cube>>
//{
//	static constexpr size_t value = number_of_points<
//			typename facet<model::Primitive<N, CS, tags::Cube> >::type>::value
//			* 2;
//};
//
//} // namespace traits
// template<typename CS>
// typename traits::length_type<CS>::type distance(
//		model::Primitive<0, CS> const & p,
//		model::Primitive<1, CS> const & line_segment)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::type distance(
//		model::Primitive<0, CS> const & p,
//		model::Primitive<2, CS, tags::Cube> const & rect)
//{
//
//}
// template<typename CS>
// typename traits::length_type<CS>::type length(
//		model::Primitive<2, CS> const & rect)
//{
//}
// template<typename CS>
// typename traits::area_type<CS>::type area(
//		model::Primitive<2, CS, tags::Cube> const & rect)
//{
//}
// template<typename CS>
// typename traits::volume_type<CS>::type volume(
//		model::Primitive<3, CS, tags::Cube> const & poly)
//{
//}
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_CUBE_H_ */
