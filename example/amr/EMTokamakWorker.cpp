//
// Created by salmon on 16-11-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
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
    void Deserialize(shared_ptr<data::DataTable> const& cfg) override;

    GEqdsk geqdsk;
    //    void InitialCondition(Real time_now) override;
    //    void ApplyBoundaryCondition(Real time_now, Real dt) override;
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
void EMTokamak::Deserialize(shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr) { return; }

    unsigned int PhiAxe = 2;
    nTuple<Real, 2> phi = cfg->GetValue("Phi", nTuple<Real, 2>{0, TWOPI});
    geqdsk.load(cfg->GetValue<std::string>("gfile", "gfile"));
    GetModel().SetObject("Limiter", std::make_shared<geometry::RevolveZ>(geqdsk.limiter(), PhiAxe, phi[0], phi[1]));
    GetModel().SetObject("Center", std::make_shared<geometry::RevolveZ>(geqdsk.boundary(), PhiAxe, phi[0], phi[1]));

    engine::Context::Initialize();
    engine::Context::Deserialize(cfg);

    GetDomain("Limiter")->AddGeoObject("Center", GetModel().GetObject("Center"));
    GetDomain("Limiter")->AddGeoObject("Antenna", GetModel().GetObject("Antenna"));

    auto amp = cfg->GetValue<Real>("antenna/amp", 1.0);
    auto n_phi = cfg->GetValue<Real>("antenna/n_phi", 1.0);
    auto omega = cfg->GetValue<Real>("antenna/omega", 1.0e9);

    typedef mesh::CylindricalSMesh mesh_type;
    auto d = GetDomain("Limiter");
    if (d != nullptr) {
        d->OnBoundaryCondition.Connect([=](Domain* self, Real time_now, Real time_dt) {
            //            auto& E = self->GetAttribute<Field<mesh_type, Real, EDGE>>("E");
            auto B = self->GetAttribute<Field<mesh_type, Real, FACE>>("B", "FULL");
            auto B0 = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("B0");

            //            E[self->GetBoundaryRange(EDGE)] = 0;
            //            B[self->GetBoundaryRange(FACE)] = 0;
            auto Bv = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("Bv");

            B.Clear();
            Bv.Clear();

            B = [&](point_type const& x) -> Vec3 { return nTuple<Real, 3>{0, 0, 1}; };
            Bv = map_to<VOLUME>(B);

            self->GetAttribute<Field<mesh_type, Real, EDGE>>("J", "Antenna") = [=](point_type const& x) -> Vec3 {
                Vec3 res{amp * std::sin(x[2]), 0, amp * std::cos(x[2])};
                res *= std::sin(n_phi * x[2]) * std::sin(omega * time_now);
                return res;
            };
        });

        d->OnInitialCondition.Connect([&](Domain* self, Real time_now) {
            auto ne = self->GetAttribute<Field<mesh_type, Real, VERTEX>>("ne", "Center");
            ne.Clear();
            ne = [&](point_type const& x) -> Real { return geqdsk.profile("ne", x[0], x[1]); };

            auto B0 = self->GetAttribute<Field<mesh_type, Real, VOLUME, 3>>("B0", "FULL");
            B0.Clear();
            B0 = [&](point_type const& x) -> Vec3 { return geqdsk.B(x[0], x[1]); };

        });
    }
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
