/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include <simpla/toolbox/Log.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/mpl/type_traits.h>
#include <simpla/toolbox/sp_def.h>
#include "GeoAlgorithm.h"

namespace simpla { namespace geometry
{
template<typename TObj> struct GeoObjectAdapter;

/**
 * @ingroup geometry
 *
 *  PlaceHolder Geometric object
 */
class GeoObject
{

    typedef GeoObject this_type;
    box_type m_bound_box_{{0, 0, 0},
                          {1, 1, 1}};
public:

    GeoObject() {}

    virtual ~GeoObject() {}

    box_type const &box() const { return m_bound_box_; };

    void box(box_type const &b) { m_bound_box_ = b; };

    /**
     * @return  check \f$ (x,y,z)\f$ in \f$ M\f$
     *           `in` then 1
     *           `out` then 0
     */
    virtual int check_inside(const Real *x) const { return geometry::in_box(m_bound_box_, x) ? 1 : 0; };

    inline int check_inside(const point_type &x) const { return check_inside(&x[0]); };

    /**
     * return id= 0b012345...
     */
    template<typename P0, typename ...Others>
    inline int check_inside(P0 const &p0, Others &&...others) const
    {
        return (check_inside(p0) << (sizeof...(others))) | check_inside(std::forward<Others>(others)...);
    };

private:

    template<typename T, size_t ...I>
    inline int check_inside_invoke_helper(T const &p_tuple, index_sequence<I...>) const
    {
        return check_inside(std::get<I>(std::forward<T>(p_tuple))...);
    };


public:
    template<typename ...Others>
    inline int check_inside(std::tuple<Others...> const &p_tuple) const
    {
        return check_inside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    };

    inline int check_inside(int num, point_type const *p_tuple) const
    {
        ASSERT(num < std::numeric_limits<int>::digits);

        int res = 0;
        for (int i = 0; i < num; ++i)
        {
            res = (res << 1) | check_inside(&p_tuple[i][0]);
        }
        return res;
    };

    virtual std::tuple<point_type, point_type, Real> nearest_point(GeoObject const &other) const
    {
        return nearest_point_box(other.box());
    };


    /**
     * @return  if  \f$ BOX \cap M \neq \emptyset \f$ then x0,x1 is set to overlap box
     *          else x0,x1 is not changed
     *         if \f$ BOX \cap M  = \emptyset \f$    return 0
     *         else if  \f$ BOX \in M   \f$ return 2
     *         else return 1
     */

    virtual std::tuple<point_type, point_type, Real> nearest_point_box(box_type const &b) const
    {
        UNIMPLEMENTED;
        return std::make_tuple(point_type{std::numeric_limits<Real>::quiet_NaN(),
                                          std::numeric_limits<Real>::quiet_NaN(),
                                          std::numeric_limits<Real>::quiet_NaN()},
                               point_type{std::numeric_limits<Real>::quiet_NaN(),
                                          std::numeric_limits<Real>::quiet_NaN(),
                                          std::numeric_limits<Real>::quiet_NaN()},
                               std::numeric_limits<Real>::quiet_NaN());
    };

    /**
     * find nearest point from \f$M\f$ to \f$x\f$
     *
     * @inout x
     * @return distance
     *  if \f$ x \in M \f$ then  distance < 0
     *  else if \f$ x \in \partial M \f$ then  distance = 0
     *  else > 0
     */
    virtual std::tuple<point_type, point_type, Real> nearest_point(Real const *x0) const
    {
        return geometry::nearest_point_to_box(m_bound_box_, x0);
    };

    virtual std::tuple<point_type, point_type, Real> nearest_point(Real const *x0, Real const *x1) const
    {
        return geometry::nearest_point_to_box(m_bound_box_, x0, x1);
    };

    virtual std::tuple<point_type, point_type, Real> nearest_point(Real const *x0, Real const *x1, Real const *x2) const
    {
        return geometry::nearest_point_to_box(m_bound_box_, x0, x1, x2);
    };

    virtual std::tuple<point_type, point_type, Real>
    nearest_point(Real const *x0, Real const *x1, Real const *x2, Real const *x3) const
    {
        return geometry::nearest_point_to_box(m_bound_box_, x0, x1, x2, x3);

    };

    template<typename ...Args>
    std::tuple<point_type, point_type, Real> nearest_point(Args &&...args) const
    {
        return nearest_point(m_bound_box_, &(args[0])...);

    };

private:

    template<typename T, size_t ...I>
    inline int nearest_point_invoke_helper(T const &p_tuple, index_sequence<I...>) const
    {
        return nearest_point(std::get<I>(std::forward<T>(p_tuple))...);
    };


public:
    template<typename ...Others>
    inline int nearest_point(std::tuple<Others...> const &p_tuple) const
    {
        return check_inside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    };


};

template<typename TObj>
struct GeoObjectAdapter : public GeoObject
{
public:

private:
};
} // namespace geometry
} // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
