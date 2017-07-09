//
// Created by salmon on 16-11-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/mesh/RectMesh.h>
#include <simpla/model/Cube.h>
#include <simpla/physics/Constants.h>
#include <iostream>

namespace simpla {

class Tokamak : public engine::Context {
    SP_OBJECT_HEAD(Tokamak, engine::Context)
   public:
    Tokamak() = default;
    ~Tokamak() override = default;

    SP_DEFAULT_CONSTRUCT(Tokamak);
    DECLARE_REGISTER_NAME(Tokamak);

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> cfg) override;
};

REGISTER_CREATOR(Tokamak)

std::shared_ptr<data::DataTable> Tokamak::Serialize() const {
    auto res = engine::Context::Serialize();
    return res;
}
void Tokamak::Deserialize(std::shared_ptr<data::DataTable> cfg) {
    engine::Context::Deserialize(cfg);

    typedef mesh::RectMesh mesh_type;

    auto d = GetDomain("Tokamak");
    if (d != nullptr) {
        //        auto geqdsk = std::dynamic_pointer_cast<GEqdsk>(GetModel().GetObject("Tokamak"));
        //        d->PreInitialCondition.Connect([=](engine::Domain* self, Real time_now) {
        //            //        if (self->check("ne", typeid(Field<mesh_type, Real, VOLUME>)))
        //            //        {
        //            //        auto& ne = self->Get("ne")->cast_as<Field<mesh_type, Real, VOLUME>>();
        //            auto ne = self->GetAttribute<Field<mesh_type, Real, VOLUME>>("ne", "Tokamak.Center");
        //            ne.Clear();
        //            ne = [=] __host__ __device__(point_type const& x) -> Real { return geqdsk->profile("ne", x[0],
        //            x[1]); };
        //            //        }
        //            //
        //            //        //        if (self->check("B0v", typeid(Field<mesh_type, Real, VOLUME, 3>)))
        //            //        {
        //
        //            //        auto& B0v = self->Get("B0v")->cast_as<Field<mesh_type, Real, VOLUME, 3>>();
        //            auto B0v = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("B0v");
        //            B0v.Clear();
        //            B0v = [=] __host__ __device__(point_type const& x) -> Vec3 { return geqdsk->B(x[0], x[1]); };
        //            //        }
        //        });
    }
}

}  // namespace simpla {
