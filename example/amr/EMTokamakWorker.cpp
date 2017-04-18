//
// Created by salmon on 16-11-29.
//

#include <simpla/utilities/sp_def.h>

#include <simpla/algebra/all.h>
#include <simpla/engine/all.h>
#include <simpla/mesh/CylindricalGeometry.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
#include <simpla/predefine/physics/EMFluid.h>
#include <iostream>

namespace simpla {
using namespace engine;
// using namespace model;

class EMTokamakWorker;

class EMTokamakWorker : public EMFluid<mesh::CylindricalGeometry> {
    SP_OBJECT_HEAD(EMTokamakWorker, EMFluid<mesh::CylindricalGeometry>)
   public:
    explicit EMTokamakWorker() : base_type() {}
    ~EMTokamakWorker() {}

    virtual void Initialize();
    virtual void Finalize();
    virtual void Advance(Real dt);
    virtual void SetPhysicalBoundaryConditions();
    virtual void SetPhysicalBoundaryConditionE();
    virtual void SetPhysicalBoundaryConditionB();

    field_type<VERTEX> psi{base_type::m_mesh_, "name"_ = "psi"};
    std::function<Vec3(point_type const &, Real)> J_src_fun;
    std::function<Vec3(point_type const &, Real)> E_src_fun;
};

void EMTokamakWorker::Initialize() {
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
void EMTokamakWorker::Finalize() {}

void EMTokamakWorker::Advance(Real dt) { base_type::Advance(dt); };

void EMTokamakWorker::SetPhysicalBoundaryConditions() {
    base_type::SetPhysicalBoundaryConditions();
    //    if (J_src_fun) {
    //        J1.Assign(model()->select(EDGE, "J_SRC"), [&](point_type const &x) -> Vec3 { return J_src_fun(x,
    //        data_time); });
    //    }
    //    if (E_src_fun) {
    //        E.Assign(model()->select(EDGE, "E_SRC"), [&](point_type const &x) -> Vec3 { return E_src_fun(x,
    //        data_time); });
    //    }
};

void EMTokamakWorker::SetPhysicalBoundaryConditionE() {
    //    E.Assign(model()->interface(EDGE, "PLASMA", "VACUUM"), 0);
}

void EMTokamakWorker::SetPhysicalBoundaryConditionB() {
    //    B.Assign(model()->interface(FACE, "PLASMA", "VACUUM"), 0);
}
}  // namespace simpla {
