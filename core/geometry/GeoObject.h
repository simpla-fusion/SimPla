/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include "../gtl/utilities/log.h"
#include "../gtl/ntuple.h"
#include "../gtl/type_traits.h"

namespace simpla { namespace geometry
{
/**
 * @ingroup geometry
 *
 *  Base Geometric object
 */
class GeoObject
{

    typedef GeoObject this_type;

public:
    typedef nTuple<Real, 3> vector_type;
    typedef nTuple<Real, 3> point_type;
    typedef nTuple<Real, 2, 3> box_type;

    GeoObject() { }

    virtual ~GeoObject() { }

    virtual box_type box() const = 0;

    /**
     * @return  check \f$ (x,y,z)\f$ in \f$ M\f$
     *           `in` then <0
     *           `on boundary` then =0
     *           `out` then >0
     */
    virtual int within(point_type const &x) const = 0;


    /**
     * return flag= 0b543210
     */
    template<typename ...Others>
    inline int within(point_type const &b, Others &&...others) const
    {
        return (within(std::forward<Others>(others)...) << 1UL) | within(b);
    };

private:

    template<typename T, int ...I>
    inline int _invoke_helper(T const &p_tuple, index_sequence<I...>) const
    {
        return within(std::get<I>(std::forward<T>(p_tuple))...);
    };


public:
    template<typename ...Others>
    inline int within(std::tuple<Others...> const &p_tuple) const
    {
        return _invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    };

    inline int within(int num, point_type const *p_tuple) const
    {
        ASSERT(num < std::numeric_limits<int>::digits);

        int res;
        for (int i = 0; i < num; ++i)
        {
            res |= (within(p_tuple[i]) > 0 ? 1 : 0) << (num - 1 - i);
        }
        return res;
    };

    /**
     * @return  if  \f$ BOX \cap M \neq \emptyset \f$ then x0,x1 is set to overlap box
     *          else x0,x1 is not changed
     *         if \f$ BOX \cap M  = \emptyset \f$    return 0
     *         else if  \f$ BOX \in M   \f$ return 2
     *         else return 1
     */
    virtual int box_intersection(point_type *x0, point_type *x1) const = 0;


    /**
     * find nearest point from \f$M\f$ to \f$x\f$
     *
     * @inout x
     * @return distance
     *  if \f$ x \in M \f$ then  distance < 0
     *  else if \f$ x \in \partial M \f$ then  distance = 0
     *  else > 0
     */
    virtual Real nearest_point(point_type *x) const = 0;


    virtual Real nearest_point(point_type *x0, point_type *x1) const
    {
        UNIMPLEMENTED;
        return std::numeric_limits<Real>::quiet_NaN();
    };

    virtual Real nearest_point(point_type const &x0, point_type const *x1, point_type *x2) const
    {
        UNIMPLEMENTED;
        return std::numeric_limits<Real>::quiet_NaN();
    };

    virtual Real nearest_point(point_type const &x0, point_type const &x1, point_type *x2,
                               point_type *x3) const
    {
        UNIMPLEMENTED;
        return std::numeric_limits<Real>::quiet_NaN();
    };

};
} // namespace geometry
} // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
