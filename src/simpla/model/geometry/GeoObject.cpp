//
// Created by salmon on 17-2-21.
//
#include "GeoObject.h"
namespace simpla {
namespace geometry {

struct GeoObject::pimpl_s {};
GeoObject::GeoObject() : m_pimpl_(new pimpl_s) {}
GeoObject::~GeoObject(){};
box_type GeoObject::bound_box() const {}
bool GeoObject::isNull() const { return true; }
bool GeoObject::isSolid() const { return false; }
bool GeoObject::isSurface() const { return false; }
bool GeoObject::isCurve() const { return false; }
}  // namespace geometry {
}  // namespace simpla {