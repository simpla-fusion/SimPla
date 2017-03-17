/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include <simpla/algebra/nTuple.h>
#include <simpla/mpl/type_traits.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/sp_def.h>
#include "GeoAlgorithm.h"

namespace simpla {
namespace geometry {
template <typename TObj>
struct GeoObjectAdapter;

/**
 * @ingroup geometry
 *
 *  PlaceHolder Geometric object
 */
class GeoObject {
    typedef GeoObject this_type;
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};

   public:
    GeoObject(){};
    GeoObject(GeoObject const &){};
    virtual ~GeoObject(){};
    virtual box_type const &bound_box() const { return m_bound_box_; };
    bool isNull() const { return true; };
    virtual bool isSolid() const { return false; };
    virtual bool isSurface() const { return false; };
    virtual bool isCurve() const { return false; };
    virtual Real distance(point_type const &x) const { return 0; }

    /**
    * @return  check \f$ (x,y,z)\f$ in \f$ M\f$
    *           `in` then 1
    *           `out` then 0
    */
    virtual int check_inside(const point_type &x) const { return in_box(bound_box(), x) ? 1 : 0; };

    int check_inside() const { return 0; }

    /**
     * return id= 0b012345...
     */
    template <typename P0, typename... Others>
    int check_inside(P0 const &p0, Others &&... others) const {
        return (check_inside(p0) << (sizeof...(others))) | check_inside(std::forward<Others>(others)...);
    };

   private:
    template <typename T, size_t... I>
    int check_inside_invoke_helper(T const &p_tuple, index_sequence<I...>) const {
        return check_inside(std::get<I>(std::forward<T>(p_tuple))...);
    };

   public:
    template <typename... Others>
    int check_inside(std::tuple<Others...> const &p_tuple) const {
        return check_inside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    };

    int check_inside(int num, point_type const *p_tuple) const {
        ASSERT(num < std::numeric_limits<int>::digits);
        int res = 0;
        for (int i = 0; i < num; ++i) { res = (res << 1) | check_inside(&p_tuple[i][0]); }
        return res;
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
    virtual std::tuple<point_type, point_type, Real> nearest_point(point_type const &x0) const {
        return geometry::nearest_point_to_box(bound_box(), x0);
    };

    virtual std::tuple<point_type, point_type, Real> nearest_point(GeoObject const &other) const {
        return nearest_point_box(other.bound_box());
    };

    /**
     * @brief  nearest point to line-segment
     * @param x0
     * @param x1
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> nearest_point(point_type const &x0, point_type const &x1) const {
        return geometry::nearest_point_to_box(bound_box(), x0, x1);
    };

    /**
     * @brief nearest point to triangle (face)
     * @param x0
     * @param x1
     * @param x2
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> nearest_point(point_type const &x0, point_type const &x1,
                                                                   point_type const &x2) const {
        return geometry::nearest_point_to_box(bound_box(), x0, x1, x2);
    };

    /**
     * @brief nearest point to Tetrahedron
     * @param x0
     * @param x1
     * @param x2
     * @param x3
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> nearest_point(point_type const &x0, point_type const &x1,
                                                                   point_type const &x2, point_type const &x3) const {
        return geometry::nearest_point_to_box(bound_box(), x0, x1, x2, x3);
    };

    /**
     * @return  if  \f$ BOX \cap M \neq \emptyset \f$ then x0,x1 is set to overlap box
     *          else x0,x1 is not changed
     *         if \f$ BOX \cap M  = \emptyset \f$    return 0
     *         else if  \f$ BOX \in M   \f$ return 2
     *         else return 1
     */

    virtual std::tuple<point_type, point_type, Real> nearest_point_box(box_type const &b) const {
        UNIMPLEMENTED;
        return std::make_tuple(
            point_type{std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN(),
                       std::numeric_limits<Real>::quiet_NaN()},
            point_type{std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN(),
                       std::numeric_limits<Real>::quiet_NaN()},
            std::numeric_limits<Real>::quiet_NaN());
    };

    //    template <typename... Args>
    //    std::tuple<point_type, point_type, Real> nearest_point(Args &&... args) const {
    //        return nearest_point(bound_box(), &(args[0])...);
    //    };
    virtual Real implicit_fun(point_type const &x) const { return 0; };
    //
    //   private:
    //    template <typename T, size_t... I>
    //    inline int nearest_point_invoke_helper(T const &p_tuple, index_sequence<I...>) const {
    //        return nearest_point(std::get<I>(std::forward<T>(p_tuple))...);
    //    };
    //
    //   public:
    //    template <typename... Others>
    //    inline int nearest_point(std::tuple<Others...> const &p_tuple) const {
    //        return check_inside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    //    };
};

template <typename U>
struct GeoObjectAdapter : public GeoObject, public U {};

class GeoObjectInverse : public GeoObject {
    GeoObject m_left_;

   public:
    GeoObjectInverse(GeoObject const &l) : m_left_(l) {}
    virtual Real implicit_fun(point_type const &x) const { return -m_left_.implicit_fun(x); }
};

class GeoObjectUnion : public GeoObject {
    GeoObject m_left_;
    GeoObject m_right_;

   public:
    GeoObjectUnion(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
    virtual Real implicit_fun(point_type const &x) const {
        return std::min(m_left_.implicit_fun(x), m_right_.implicit_fun(x));
    }
};

class GeoObjectIntersection : public GeoObject {
    GeoObject m_left_;
    GeoObject m_right_;

   public:
    GeoObjectIntersection(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
    virtual Real implicit_fun(point_type const &x) const {
        return std::max(m_left_.implicit_fun(x), m_right_.implicit_fun(x));
    }
};
class GeoObjectDifference : public GeoObject {
    GeoObject m_left_;
    GeoObject m_right_;

   public:
    GeoObjectDifference(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
    virtual Real implicit_fun(point_type const &x) const {
        return std::max(m_left_.implicit_fun(x), -m_right_.implicit_fun(x));
    }
};

inline GeoObjectInverse operator-(GeoObject const &l) { return GeoObjectInverse(l); }
inline GeoObjectInverse operator!(GeoObject const &l) { return GeoObjectInverse(l); }
inline GeoObjectUnion operator+(GeoObject const &l, GeoObject const &r) { return GeoObjectUnion(l, r); }
inline GeoObjectDifference operator-(GeoObject const &l, GeoObject const &r) { return GeoObjectDifference(l, r); }
inline GeoObjectIntersection operator&(GeoObject const &l, GeoObject const &r) { return GeoObjectIntersection(l, r); }
}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
