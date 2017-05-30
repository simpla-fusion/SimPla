//
// Created by salmon on 16-11-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/geometry/Cube.h>
#include <simpla/mesh/CylindricalGeometry.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/third_part/SAMRAITimeIntegrator.h>
#include <simpla/utilities/sp_def.h>
#include <iostream>
namespace simpla {
using namespace engine;
static bool s_RegisterDomain =
    engine::Domain::RegisterCreator<EMFluid<mesh::CylindricalSMesh>>(std::string("EMFluidCylindricalSMesh"));
REGISTER_CREATOR(SAMRAITimeIntegrator)

class Tokamak : public engine::Context {
    SP_OBJECT_HEAD(Tokamak, engine::Context)
   public:
    Tokamak() = default;
    ~Tokamak() override = default;

    SP_DEFAULT_CONSTRUCT(Tokamak);
    DECLARE_REGISTER_NAME("Tokamak");

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    GEqdsk geqdsk;
};

REGISTER_CREATOR(Tokamak)

std::shared_ptr<data::DataTable> Tokamak::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue<std::string>("Type", "Tokamak");
    res->Set(Context::Serialize());
    return res;
};
void Tokamak::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr) { return; }
    engine::Context::Initialize();

    typedef mesh::CylindricalSMesh mesh_type;

    //    unsigned int PhiAxe = 2;
    //    nTuple<Real, 2> phi = cfg->GetValue("Phi", nTuple<Real, 2>{0, TWOPI});
    //    geqdsk.load(cfg->GetValue<std::string>("gfile", "gfile"));
    //    GetModel().SetObject("Limiter", std::make_shared<geometry::RevolveZ>(geqdsk.limiter(), PhiAxe, phi[0],
    //    phi[1]));
    //    GetModel().SetObject("Center", std::make_shared<geometry::RevolveZ>(geqdsk.boundary(), PhiAxe, phi[0],
    //    phi[1]));

    Vec3 amp = cfg->GetValue<nTuple<Real, 3>>("Antenna/amp", nTuple<Real, 3>{0, 0, 1});
    auto n_phi = cfg->GetValue<Real>("Antenna/n_phi", 1.0);
    auto freq = cfg->GetValue<Real>("Antenna/Frequency", 1.0e9);

    box_type antenna_box{{1.4, -0.5, -3.1415926 / 2}, {1.45, 0.5, 3.1415926 / 2}};
    std::get<0>(antenna_box) = cfg->GetValue<point_type>("Antenna/x_lower", std::get<0>(antenna_box));
    std::get<1>(antenna_box) = cfg->GetValue<point_type>("Antenna/x_upper", std::get<1>(antenna_box));

    GetModel().SetObject("Antenna", std::make_shared<geometry::Cube>(antenna_box));

    index_box_type idx_box{{0, 0, 0}, {1, 1, 1}};
    std::get<1>(idx_box) = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{64, 64, 32});
    //    GetAtlas().SetIndexBox(idx_box);

    engine::Context::Deserialize(cfg);
    auto d = GetDomain("Main");

    ASSERT(d != nullptr);
    d->SetGeoObject(GetModel().GetObject("Limiter"));
    d->SetGeoObject(GetModel().GetObject("Center"));
    d->SetGeoObject(GetModel().GetObject("Antenna"));

    //        d->OnBoundaryCondition.Connect([=](Domain* self, Real time_now, Real time_dt) {});
    //

    d->PreInitialCondition.Connect([&](Domain* self, Real time_now) {
        if (self->GetMesh()->check("ne", typeid(Field<mesh_type, Real, VOLUME>))) {
            auto ne = self->GetAttribute<Field<mesh_type, Real, VOLUME>>("ne", "Center");
            ne.Clear();
            ne = [&](point_type const& x) -> Real { return geqdsk.profile("ne", x[0], x[1]); };
        }

        if (self->GetMesh()->check("B0v", typeid(Field<mesh_type, Real, VOLUME, 3>))) {
            auto B0v = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("B0v");
            B0v.Clear();
            B0v = [&](point_type const& x) -> Vec3 { return geqdsk.B(x[0], x[1]); };
        }
    });

    d->PreAdvance.Connect([=](Domain* self, Real time_now, Real time_dt) {

        if (self->GetMesh()->check("J", typeid(Field<mesh_type, Real, EDGE>))) {
            auto J = self->GetAttribute<Field<mesh_type, Real, EDGE>>("J", "Antenna");
            J.Clear();
            J = [&](point_type const& x) -> Vec3 {
                Real a = std::sin(n_phi * x[2] + TWOPI * freq * time_now);
                return Vec3{std::sin(x[2]) * a, 0, a * std::cos(x[2])};
            };
        }

    });
    //    d->PostAdvance.Connect([=](Domain* self, Real time_now, Real time_dt) {
    //
    //        // for VisIt dump
    //        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpJ").DeepCopy(
    //            self->GetAttribute<Field<mesh_type, Real, EDGE>>("J"));
    //        //
    //        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpE").DeepCopy(
    //            self->GetAttribute<Field<mesh_type, Real, EDGE>>("E"));
    //
    //        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpB").DeepCopy(
    //            self->GetAttribute<Field<mesh_type, Real, FACE>>("B"));
    //
    //    });
}

}  // namespace simpla {
