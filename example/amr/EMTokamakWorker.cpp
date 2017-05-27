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
#include <simpla/utilities/sp_def.h>
#include <iostream>

namespace simpla {
using namespace engine;
static bool s_RegisterDomain =
    engine::Domain::RegisterCreator<EMFluid<mesh::CylindricalSMesh>>(std::string("EMFluidCylindricalSMesh"));
class EMTokamak : public engine::Context {
    SP_OBJECT_HEAD(EMTokamak, engine::Context)
   public:
    EMTokamak() = default;
    ~EMTokamak() override = default;

    SP_DEFAULT_CONSTRUCT(EMTokamak);
    DECLARE_REGISTER_NAME("EMTokamak");

    std::shared_ptr<data::DataTable> Serialize() const override;
    void Deserialize(std::shared_ptr<data::DataTable> const& cfg) override;

    GEqdsk geqdsk;
    //    void InitialCondition(Real time_now) override;
    //    void DoBoundaryCondition(Real time_now, Real dt) override;
    //    void DoAdvance(Real time_now, Real dt) override;
    //    field_type<VERTEX> psi{base_type::m_mesh_, "name"_ = "psi"};
    //    std::function<Vec3(point_type const&, Real)> J_src_fun;
    //    std::function<Vec3(point_type const&, Real)> E_src_fun;
};

REGISTER_CREATOR(EMTokamak)

std::shared_ptr<data::DataTable> EMTokamak::Serialize() const {
    auto res = std::make_shared<data::DataTable>();
    res->SetValue<std::string>("Type", "EMTokamak");
    res->Set(Context::Serialize());
    return res;
};
void EMTokamak::Deserialize(std::shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr) { return; }
    engine::Context::Initialize();

    typedef mesh::CylindricalSMesh mesh_type;

    unsigned int PhiAxe = 2;
    nTuple<Real, 2> phi = cfg->GetValue("Phi", nTuple<Real, 2>{0, TWOPI});
    geqdsk.load(cfg->GetValue<std::string>("gfile", "gfile"));
    GetModel().SetObject("Limiter", std::make_shared<geometry::RevolveZ>(geqdsk.limiter(), PhiAxe, phi[0], phi[1]));
    GetModel().SetObject("Center", std::make_shared<geometry::RevolveZ>(geqdsk.boundary(), PhiAxe, phi[0], phi[1]));

    Vec3 amp = cfg->GetValue<nTuple<Real, 3>>("Antenna/amp", nTuple<Real, 3>{0, 0, 1});
    auto n_phi = cfg->GetValue<Real>("Antenna/n_phi", 1.0);
    auto freq = cfg->GetValue<Real>("Antenna/Frequency", 1.0e9);

    box_type antenna_box{{1.4, -0.5, -3.1415926 / 2}, {1.45, 0.5, 3.1415926 / 2}};
    std::get<0>(antenna_box) = cfg->GetValue<point_type>("Antenna/x_lower", std::get<0>(antenna_box));
    std::get<1>(antenna_box) = cfg->GetValue<point_type>("Antenna/x_upper", std::get<1>(antenna_box));

    GetModel().SetObject("Antenna", std::make_shared<geometry::Cube>(antenna_box));

    index_box_type idx_box{{0, 0, 0}, {1, 1, 1}};
    std::get<1>(idx_box) = cfg->GetValue<nTuple<int, 3>>("Dimensions", nTuple<int, 3>{64, 64, 32});
    GetAtlas().SetIndexBox(idx_box);

    engine::Context::Deserialize(cfg);
    auto d = GetDomain("Main");

    ASSERT(d != nullptr);

    d->AddGeoObject("PEC", GetModel().GetObject("Limiter"));
    d->AddGeoObject("Center", GetModel().GetObject("Center"));
    d->AddGeoObject("Antenna", GetModel().GetObject("Antenna"));

    //        d->OnBoundaryCondition.Connect([=](Domain* self, Real time_now, Real time_dt) {});
    //
    d->PreInitialCondition.Connect([&](Domain* self, Real time_now) {
        auto ne = self->GetAttribute<Field<mesh_type, Real, VOLUME>>("ne", "Center");
        ne.Clear();
        ne = [&](point_type const& x) -> Real { return geqdsk.profile("ne", x[0], x[1]); };

        auto B0v = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("B0v");
        B0v.Clear();
        B0v = [&](point_type const& x) -> Vec3 { return geqdsk.B(x[0], x[1]); };
    });

    d->PreAdvance.Connect([=](Domain* self, Real time_now, Real time_dt) {
        auto J = self->GetAttribute<Field<mesh_type, Real, EDGE>>("J", "Antenna");
        J.Clear();
        J = [&](point_type const& x) -> Vec3 {
            Real a = std::sin(n_phi * x[2] + TWOPI * freq * time_now);
            return Vec3{std::sin(x[2]) * a, 0, a * std::cos(x[2])};
        };

        //        auto Jv = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("Jv");
        //        Jv = [&](point_type const& x) -> Vec3 {
        //            Real a = std::sin(n_phi * x[2] + TWOPI * freq * time_now);
        //            return Vec3{std::sin(x[2]) * a, 0, a * std::cos(x[2])};
        //        };

    });
    d->PostAdvance.Connect([=](Domain* self, Real time_now, Real time_dt) {

        // for VisIt dump
        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpJ").DeepCopy(
            self->GetAttribute<Field<mesh_type, Real, EDGE>>("J"));
        //
        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpE").DeepCopy(
            self->GetAttribute<Field<mesh_type, Real, EDGE>>("E"));

        self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("dumpB").DeepCopy(
            self->GetAttribute<Field<mesh_type, Real, FACE>>("B"));

    });
}
//    std::cout << "Model = ";
//    GetModel().Serialize(std::cout, 0);
//    auto const &boundary = geqdsk.boundary();
//    ne.Assign([&](point_type const &x) -> Real { return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0;
//    });
//    psi.Assign([&](point_type const &x) -> Real { return geqdsk.psi(x); });
//    nTuple<Real, 3> ZERO_V{0, 0, 0};
//    //    B0.Assign([&](point_type const &x) -> Vec3 { return (geqdsk.in_limiter(x)) ? geqdsk.B(x) : ZERO_V; });
//    for (auto &item : GetSpecies()) {
//        Real ratio = db()->GetValue("Particles." + item.first + ".ratio", 1.0);
//        *item.second->rho = ne * ratio;
//    }
//
// void EMTokamak::SetPhysicalBoundaryConditions() {
//    base_type::SetPhysicalBoundaryConditions();
//    //    if (J_src_fun) {
//    //        J1.Assign(model()->select(EDGE, "J_SRC"), [&](point_type const &x) -> Vec3 { return J_src_fun(x,
//    //        data_time); });
//    //    }
//    //    if (E_src_fun) {
//    //        E.Assign(model()->select(EDGE, "E_SRC"), [&](point_type const &x) -> Vec3 { return E_src_fun(x,
//    //        data_time); });
//    //    }
//};
//
// void EMTokamak::SetPhysicalBoundaryConditionE() {
//    //    E.Assign(model()->interface(EDGE, "PLASMA", "VACUUM"), 0);
//}
//
// void EMTokamak::SetPhysicalBoundaryConditionB() {
//    //    B.Assign(model()->interface(FACE, "PLASMA", "VACUUM"), 0);
//}
}  // namespace simpla {
