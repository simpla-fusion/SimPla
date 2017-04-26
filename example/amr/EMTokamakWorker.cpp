//
// Created by salmon on 16-11-29.
//

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/mesh/all.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <simpla/utilities/sp_def.h>
#include <iostream>
namespace simpla {
using namespace engine;
// using namespace model;

class EMTokamak : public EMFluid<mesh::CylindricalSMesh> {
    SP_OBJECT_HEAD(EMTokamak, EMFluid<mesh::CylindricalSMesh>)
   public:
    DOMAIN_HEAD(EMTokamak, EMFluid<mesh::CylindricalSMesh>)

    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = std::make_shared<data::DataTable>();
        res->SetValue<std::string>("Type", "EMTokamak");
        return res;
    };

    void Deserialize(shared_ptr<DataTable> t) override { UNIMPLEMENTED; }

    void Initialize() override;
    void Finalize() override;
    void SetUp() override;
    void TearDown() override;

    void InitializeCondition(Real time_now) override;
    void BoundaryCondition(Real time_now, Real dt) override;
    void Advance(Real time_now, Real dt) override;

    field_type<VERTEX> psi{base_type::m_mesh_, "name"_ = "psi"};
    std::function<Vec3(point_type const &, Real)> J_src_fun;
    std::function<Vec3(point_type const &, Real)> E_src_fun;
};
bool EMTokamak::is_registered = engine::Domain::RegisterCreator<EMTokamak>("EMTokamak");

void EMTokamak::Initialize() {
    base_type::Initialize();
    //    rho0.Assign([&](point_type const &x) -> Real { return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0;
    //    });
    //    psi.Assign([&](point_type const &x) -> Real { return geqdsk.psi(x); });

    //    nTuple<Real, 3> ZERO_V{0, 0, 0};
    //    //    B0.Assign([&](point_type const &x) -> Vec3 { return (geqdsk.in_limiter(x)) ? geqdsk.B(x) : ZERO_V; });
    //    for (auto &item : GetSpecies()) {
    ////        Real ratio = db()->GetValue("Particles." + item.first + ".ratio", 1.0);
    //        *item.second->rho = rho0 * ratio;
    //    }
}
void EMTokamak::Finalize() { base_type::Finalize(); }
void EMTokamak::SetUp() { base_type::SetUp(); };
void EMTokamak::TearDown() { base_type::TearDown(); };

void EMTokamak::InitializeCondition(Real time_now){};
void EMTokamak::BoundaryCondition(Real time_now, Real dt){};
void EMTokamak::Advance(Real time_now, Real dt) { base_type::Advance(time_now, dt); };
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
