/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include <simpla/algebra/nTuple.h>
#include <simpla/design_pattern/design_pattern.h>
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
class GeoObject : public concept::Printable {
    SP_OBJECT_BASE(GeoObject)

   public:
    GeoObject(){};
    GeoObject(GeoObject const &){};
    virtual ~GeoObject(){};
    box_type m_bound_box_{{0, 0, 0}, {1, 1, 1}};
    virtual box_type const &GetBoundBox() const { return m_bound_box_; };
    bool isNull() const { return true; };
    virtual bool isSolid() const { return false; };
    virtual bool isSurface() const { return false; };
    virtual bool isCurve() const { return false; };
    virtual bool equal(GeoObject const &other) const { return this == &other; }
    virtual Real GetDistanceTo(point_type const &x) const { return 0; }

    bool operator==(GeoObject const &other) const { return equal(other); }
    virtual bool CheckOverlap(box_type const &b) const { return geometry::CheckOverlap(GetBoundBox(), b); }
    /**
    * @return  check \f$ (x,y,z)\f$ in \f$ M\f$
    *           `in` then 1
    *           `out` then 0
    */
    virtual int CheckInside(const point_type &x) const { return in_box(GetBoundBox(), x) ? 1 : 0; };

    int CheckInside() const { return 0; }

    /**
     * return id= 0b012345...
     */
    template <typename P0, typename... Others>
    int CheckInside(P0 const &p0, Others &&... others) const {
        return (CheckInside(p0) << (sizeof...(others))) | CheckInside(std::forward<Others>(others)...);
    };

   private:
    template <typename T, size_t... I>
    int CheckInside_invoke_helper(T const &p_tuple, index_sequence<I...>) const {
        return CheckInside(std::get<I>(std::forward<T>(p_tuple))...);
    };

   public:
    template <typename... Others>
    int CheckInside(std::tuple<Others...> const &p_tuple) const {
        return CheckInside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    };

    int CheckInside(int num, point_type const *p_tuple) const {
        ASSERT(num < std::numeric_limits<int>::digits);
        int res = 0;
        for (int i = 0; i < num; ++i) { res = (res << 1) | CheckInside(&p_tuple[i][0]); }
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
    virtual std::tuple<point_type, point_type, Real> GetNearestPoint(point_type const &x0) const {
        return geometry::GetNearestPointToBox(GetBoundBox(), x0);
    };

    virtual std::tuple<point_type, point_type, Real> GetNearestPoint(GeoObject const &other) const {
        return GetNearestPointToBox(other.GetBoundBox());
    };

    /**
     * @brief  nearest point to line-segment
     * @param x0
     * @param x1
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> GetNearestPoint(point_type const &x0, point_type const &x1) const {
        return geometry::GetNearestPointToBox(GetBoundBox(), x0, x1);
    };

    /**
     * @brief nearest point to triangle (face)
     * @param x0
     * @param x1
     * @param x2
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> GetNearestPoint(point_type const &x0, point_type const &x1,
                                                                     point_type const &x2) const {
        return geometry::GetNearestPointToBox(GetBoundBox(), x0, x1, x2);
    };

    /**
     * @brief nearest point to Tetrahedron
     * @param x0
     * @param x1
     * @param x2
     * @param x3
     * @return
     */
    virtual std::tuple<point_type, point_type, Real> GetNearestPoint(point_type const &x0, point_type const &x1,
                                                                     point_type const &x2, point_type const &x3) const {
        return geometry::GetNearestPointToBox(GetBoundBox(), x0, x1, x2, x3);
    };

    /**
     * @return  if  \f$ BOX \cap M \neq \emptyset \f$ then x0,x1 is set to overlap box
     *          else x0,x1 is not changed
     *         if \f$ BOX \cap M  = \emptyset \f$    return 0
     *         else if  \f$ BOX \in M   \f$ return 2
     *         else return 1
     */

    virtual std::tuple<point_type, point_type, Real> GetNearestPointToBox(box_type const &b) const {
        UNIMPLEMENTED;
        return std::make_tuple(
            point_type{std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN(),
                       std::numeric_limits<Real>::quiet_NaN()},
            point_type{std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN(),
                       std::numeric_limits<Real>::quiet_NaN()},
            std::numeric_limits<Real>::quiet_NaN());
    };

    //    template <typename... Args>
    //    std::tuple<point_type, point_type, Real> GetNearestPoint(Args &&... args) const {
    //        return GetNearestPoint(GetBoundBox(), &(args[0])...);
    //    };
    virtual Real implicit_fun(point_type const &x) const { return 0; };
    //
    //   private:
    //    template <typename T, size_t... I>
    //    inline int GetNearestPoint_invoke_helper(T const &p_tuple, index_sequence<I...>) const {
    //        return GetNearestPoint(std::get<I>(std::forward<T>(p_tuple))...);
    //    };
    //
    //   public:
    //    template <typename... Others>
    //    inline int GetNearestPoint(std::tuple<Others...> const &p_tuple) const {
    //        return CheckInside_invoke_helper(p_tuple, make_index_sequence<sizeof...(Others)>());
    //    };
};
//
// template <typename U>
// struct GeoObjectAdapter : public GeoObject, public U {};
//
// class GeoObjectInverse : public GeoObject {
//    GeoObject const &m_left_;
//
//   public:
//    explicit GeoObjectInverse(GeoObject const &l) : m_left_(l) {}
//    virtual Real implicit_fun(point_type const &x) const { return -m_left_.implicit_fun(x); }
//};
//
// class GeoObjectUnion : public GeoObject {
//    GeoObject const &m_left_;
//    GeoObject const &m_right_;
//
//   public:
//    GeoObjectUnion(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
//    virtual Real implicit_fun(point_type const &x) const {
//        return std::min(m_left_.implicit_fun(x), m_right_.implicit_fun(x));
//    }
//};
//
// class GeoObjectIntersection : public GeoObject {
//    GeoObject const &m_left_;
//    GeoObject const &m_right_;
//
//   public:
//    GeoObjectIntersection(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
//    virtual Real implicit_fun(point_type const &x) const {
//        return std::max(m_left_.implicit_fun(x), m_right_.implicit_fun(x));
//    }
//};
// class GeoObjectDifference : public GeoObject {
//    GeoObject const &m_left_;
//    GeoObject const &m_right_;
//
//   public:
//    GeoObjectDifference(GeoObject const &l, GeoObject const &r) : m_left_(l), m_right_(r) {}
//    virtual Real implicit_fun(point_type const &x) const {
//        return std::max(m_left_.implicit_fun(x), -m_right_.implicit_fun(x));
//    }
//};
//
// inline GeoObjectInverse operator-(GeoObject const &l) { return GeoObjectInverse(l); }
// inline GeoObjectInverse operator!(GeoObject const &l) { return GeoObjectInverse(l); }
// inline GeoObjectUnion operator+(GeoObject const &l, GeoObject const &r) { return GeoObjectUnion(l, r); }
// inline GeoObjectDifference operator-(GeoObject const &l, GeoObject const &r) { return GeoObjectDifference(l, r); }
// inline GeoObjectIntersection operator&(GeoObject const &l, GeoObject const &r) { return GeoObjectIntersection(l, r);
// }

struct GeoObjectFactory {
    GeoObjectFactory();
    ~GeoObjectFactory();

    bool RegisterCreator(std::string const &k,
                         std::function<std::shared_ptr<GeoObject>(std::shared_ptr<data::DataEntity> const &)> const &);

    template <typename U>
    bool RegisterCreator(std::string const &k) {
        RegisterCreator(k, [&](std::shared_ptr<data::DataEntity> const &t) { return std::make_shared<U>(t); });
    }

    std::shared_ptr<GeoObject> Create(std::shared_ptr<data::DataEntity> const &p) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

#define GLOBAL_GEO_OBJECT_FACTORY SingletonHolder<simpla::geometry::GeoObjectFactory>::instance()

}  // namespace geometry
namespace data {
template <typename U>
struct data_entity_traits<U, std::enable_if_t<std::is_base_of<geometry::GeoObject, U>::value>> {
    static U from(DataEntity const &v) { return v.cast_as<DataEntityWrapper<U>>().value(); };
    static std::shared_ptr<DataEntity> to(U const &v) {
        auto t = std::make_shared<DataTable>();
        t->SetValue("type", v.GetClassName());
        t->SetValue("bound_box", v.GetBoundBox());
        return std::dynamic_pointer_cast<DataEntity>(t);
    };
};
}
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
