//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include "simpla/SIMPLA_config.h"

#include <functional>

#include "simpla/geometry/GeoObject.h"
#include "simpla/utilities/Factory.h"

#include "Attribute.h"
#include "EngineObject.h"
namespace simpla {

namespace engine {
using namespace data;

class Model : public EngineObject {
    SP_OBJECT_HEAD(Model, EngineObject)

   public:
    void DoSetUp() override;
    void DoUpdate() override;
    void DoTearDown() override;

    std::shared_ptr<Model> GetParent() const;
    void SetParent(std::shared_ptr<Model> const &);

    box_type const &GetBoundingBox() const;

    typedef std::function<Real(point_type const &)> attr_fun;
    typedef std::function<Vec3(point_type const &)> vec_attr_fun;
    virtual attr_fun GetAttribute(std::string const &attr_name) const { return nullptr; };
    virtual vec_attr_fun GetAttributeVector(std::string const &attr_name) const { return nullptr; };

    void Load(std::string const &url);

    virtual std::shared_ptr<geometry::GeoObject> Get(std::string const &k) const;
    virtual size_type Delete(std::string const &k);
    virtual size_type Add(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);

    virtual size_type AddAttribute(std::string const &model_name, std::string const &function_name, attr_fun);
    virtual size_type AddAttribute(std::string const &model_name, std::string const &function_name, vec_attr_fun);

    //    void SetObject(std::string const &k, std::shared_ptr<geometry::GeoObject> const &);
    //    std::shared_ptr<geometry::GeoObject> GetGeoObject(std::string const &k) const;
    //    size_type DeleteObject(std::string const &);
    //    std::map<std::string, std::shared_ptr<geometry::GeoObject>> const &GetAll() const;

    void LoadAttribute(std::string const &k, Attribute *f) const;

    template <typename TV, int IFORM>
    void LoadAttribute(std::string const &k, AttributeT<TV, IFORM> *f) const {
        int n = ((IFORM == NODE || IFORM == CELL) ? 1 : 3) * f->GetDOF();
        if (n == 1) {
            auto fun = GetAttribute(k);
            //            if (fun) { *f = fun; }
        } else {
            auto fun = GetAttributeVector(k);
            //            if (fun) { *f = fun; }
        }
    };
};
}  // namespace geometry {
}  // namespace simpla{namespace geometry{

#endif  // SIMPLA_MODEL_H
