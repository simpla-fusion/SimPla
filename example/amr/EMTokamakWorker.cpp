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
static bool s_RegisterDomain =
    engine::Domain::RegisterCreator<EMFluid<mesh::Mesh<mesh::CylindricalGeometry, mesh::SMesh>>>(
        "EMFluidCylindricalSMesh");
class EMTokamak : public engine::Context {
    SP_OBJECT_HEAD(EMTokamak, engine::Context)
   public:
    EMTokamak() = default;
    ~EMTokamak() override = default;

    SP_DEFAULT_CONSTRUCT(EMTokamak);
    DECLARE_REGISTER_NAME("EMTokamak");

    //    DOMAIN_HEAD(EMTokamak, EMFluid<mesh::CylindricalSMesh>)

    void Initialize() override;
    void Finalize() override;
    std::shared_ptr<data::DataTable> Serialize() const override {
        auto res = std::make_shared<data::DataTable>();
        res->SetValue<std::string>("Type", "EMTokamak");
        return res;
    };

    void Deserialize(shared_ptr<data::DataTable> const& cfg) override;

    //    void InitializeCondition(Real time_now) override;
    //    void BoundaryCondition(Real time_now, Real dt) override;
    //    void Advance(Real time_now, Real dt) override;
    //    field_type<VERTEX> psi{base_type::m_mesh_, "name"_ = "psi"};
    std::function<Vec3(point_type const&, Real)> J_src_fun;
    std::function<Vec3(point_type const&, Real)> E_src_fun;
};

REGISTER_CREATOR(EMTokamak)

void EMTokamak::Initialize() {}
void EMTokamak::Finalize() {}
void EMTokamak::Deserialize(shared_ptr<data::DataTable> const& cfg) {
    if (cfg == nullptr) { return; }

    GEqdsk geqdsk;

    Real phi0 = 0, phi1 = TWOPI;
    geqdsk.load(cfg->GetValue<std::string>("gfile", "gfile"));
    GetModel().SetObject("Boundary", std::make_shared<geometry::RevolveZ>(geqdsk.boundary(), 2, 0, TWOPI));
    GetModel().SetObject("Boundary", std::make_shared<geometry::RevolveZ>(geqdsk.limiter(), 2, 0, TWOPI));

    cfg->GetTable("Domains")->Foreach([&](std::string const& k, std::shared_ptr<data::DataEntity> const v) {
        Context::SetDomain(k, Domain::Create(v, GetModel().GetObject(k)));
    });
    //    std::cout << "Model = ";
    //    GetModel().Serialize(std::cout, 0);
    //
    //    auto const &boundary = geqdsk.boundary();
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
