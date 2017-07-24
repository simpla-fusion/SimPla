/**
 * @file GeoObject.h
 *
 *  Created on: 2015-6-7
 *      Author: salmon
 */

#ifndef CORE_GEOMETRY_GEO_OBJECT_H_
#define CORE_GEOMETRY_GEO_OBJECT_H_

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/nTuple.ext.h"
#include "simpla/algebra/nTuple.h"
#include "simpla/data/Serializable.h"
#include "simpla/engine/EnableCreateFromDataTable.h"
#include "simpla/engine/SPObject.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/SPDefines.h"
#include "simpla/utilities/type_traits.h"

#include "BoxUtilities.h"

namespace simpla {
namespace geometry {

enum GeometryType { SP_NULL_GEO, SP_SOLID, SP_CURVE, SP_SURFACE, SP_POINT };
/**
 * @ingroup geometry
 *
 *  PlaceHolder Geometric object
 */
class GeoObject : public engine::EnableCreateFromDataTable<GeoObject> {
    SP_OBJECT_HEAD(GeoObject, engine::EnableCreateFromDataTable<GeoObject>)
    SP_DEFAULT_CONSTRUCT(GeoObject)

   public:
    GeoObject() = default;
    ~GeoObject() override = default;

    std::shared_ptr<data::DataTable> Serialize() const override { return base_type::Serialize(); };
    void Deserialize(std::shared_ptr<data::DataTable> const &t) override { base_type::Deserialize(t); }

    bool operator==(GeoObject const &other) const { return equal(other); }

    virtual int GetGeoType() const { return SP_SOLID; }

    virtual bool isNull() const { return true; };
    virtual bool isFull() const { return false; };

    virtual bool equal(GeoObject const &other) const { return this == &other; }

    virtual Real measure() const { return Measure(BoundingBox()); };

    virtual box_type BoundingBox() const { return box_type{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}; }

    /**
    * @brief
    * @param b
    * @return  volume(*this & b)/ volume(*this)
    *         1   all in
    *          0<res<1  intersection
    *          0   out-side
    */
    virtual Real CheckOverlap(box_type const &b) const { return Measure(Overlap(BoundingBox(), b)) / measure(); }

    virtual Real CheckOverlap(GeoObject const &other) const { return CheckOverlap(other.BoundingBox()); }
    /**
    * @return  check \f$ (x,y,z)\f$ in \f$ M\f$
    *           `in` then 1
    *           `out` then 0
    */
    virtual bool CheckInside(const point_type &x) const { return CheckInSide(BoundingBox(), x); };

    virtual std::tuple<Real, point_type, point_type> Distance(point_type const &x) const {
        return std::tuple<Real, point_type, point_type>{0, x, x};
    }
    //    virtual std::tuple<Real, point_type, point_type> Distance(box_type const &b) const {
    //        return NearestPoint(BoundingBox(), b);
    //    }
    //    virtual std::tuple<Real, point_type, point_type> Distance(GeoObject const &other) const {
    //        return NearestPoint(BoundingBox(), other.BoundingBox());
    //    }
};

}  // namespace geometry
}  // namespace simpla

#endif /* CORE_GEOMETRY_GEO_OBJECT_H_ */
