//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include "simpla/SIMPLA_config.h"

#include <functional>

#include "simpla/algebra/Field.h"
#include "simpla/geometry/GeoObject.h"

#include "EngineObject.h"
#include "simpla/utilities/Factory.h"
namespace simpla {

namespace engine {
using namespace data;

class Model : public EngineObject {
    SP_OBJECT_HEAD(Model, EngineObject)

    Model(std::string const &);

    void DoInitialize() override;
    void DoUpdate() override;
    void DoTearDown() override;
    void DoFinalize() override;

    box_type const &BoundingBox() const;

    typedef std::function<Real(point_type const &)> attr_fun;
    typedef std::function<Vec3(point_type const &)> vec_attr_fun;

    virtual attr_fun GetAttribute(std::string const &attr_name) const { return nullptr; };
    virtual vec_attr_fun GetAttributeVector(std::string const &attr_name) const { return nullptr; };

    void SetObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);
    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::string const &k) const;
    size_type DeleteObject(std::string const &);

    std::map<std::string, std::shared_ptr<geometry::GeoObject>> const &GetAll() const;

    template <typename TD, typename TV, int IFORM, int... N>
    void LoadProfile(std::string const &k, Field<TD, TV, IFORM, N...> *f) const {
        int n = ((IFORM == NODE || IFORM == CELL) ? 1 : 3) * simpla::reduction_v(tags::multiplication(), 1, N...);
        if (n == 1) {
            auto fun = GetAttribute(k);
            if (fun) { *f = fun; }
        } else {
            auto fun = GetAttributeVector(k);
            if (fun) { *f = fun; }
        }
    };
};
}  // namespace geometry {
}  // namespace simpla{namespace geometry{

#endif  // SIMPLA_MODEL_H
