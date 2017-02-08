//
// Created by salmon on 16-11-29.
//

#include <simpla/SIMPLA_config.h>

#include <simpla/algebra/all.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/mesh/Worker.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
#include <simpla/predefine/CalculusPolicy.h>
#include <simpla/predefine/CartesianGeometry.h>
#include <simpla/predefine/CylindricalGeometry.h>
#include <iostream>
#include "../../scenario/problem_domain/EMFluid.h"

namespace simpla {
using namespace mesh;
// using namespace model;

class EMTokamakWorker;
std::shared_ptr<mesh::MeshView> create_mesh() { return std::make_shared<mesh::CylindricalGeometry>(); }

std::shared_ptr<mesh::Worker> create_worker() {
    return std::dynamic_pointer_cast<mesh::Worker>(std::make_shared<EMTokamakWorker>());
}

class EMTokamakWorker : public EMFluid<mesh::CylindricalGeometry> {
   public:
    SP_OBJECT_HEAD(EMTokamakWorker, EMFluid<mesh::CylindricalGeometry>);
    explicit EMTokamakWorker() : base_type() {}
    ~EMTokamakWorker() {}

    virtual std::shared_ptr<MeshView> create_mesh() { return std::make_shared<mesh_type>(this); };
    virtual void Initialize();
    virtual void PreProcess();
    virtual void PostProcess();
    virtual void Finalize();
    virtual void Initialize(Real data_time);
    virtual void Finalize(Real data_time);

    virtual void NextTimeStep(Real data_time, Real dt);

    virtual void SetPhysicalBoundaryConditions(Real data_time);

    virtual void SetPhysicalBoundaryConditionE(Real time);

    virtual void SetPhysicalBoundaryConditionB(Real time);

    GEqdsk geqdsk;

    field_type<VERTEX> psi{this, {"name"_ = "psi"}};

    std::function<Vec3(point_type const &, Real)> J_src_fun;

    std::function<Vec3(point_type const &, Real)> E_src_fun;
};

void EMTokamakWorker::Initialize() {
    if (TryInitialize()) { return; };
    base_type::Initialize();

    // first run, only Load configure, m_mesh_=nullptr
    geqdsk.load(db.getValue("GEqdsk", "geqdsk.gfile"));

    db.asTable("Particles").foreach ([&](std::string const &key, data::DataEntity const &item) {
        add_particle(key, item.asTable());
    });

    db.setValue("bound_box", geqdsk.box());

    model()->AddObject("VACUUM", geqdsk.limiter_gobj());

    model()->AddObject("PLASMA", geqdsk.boundary_gobj());

};

void EMTokamakWorker::PreProcess() { base_type::PreProcess(); }
void EMTokamakWorker::PostProcess() { base_type::PostProcess(); }

void EMTokamakWorker::Initialize(Real data_time) {
    PreProcess();

    rho0.Assign([&](point_type const &x) -> Real { return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0; });

    psi.Assign([&](point_type const &x) -> Real { return geqdsk.psi(x); });

    nTuple<Real, 3> ZERO_V{0, 0, 0};

    B0.Assign([&](point_type const &x) -> Vec3 { return (geqdsk.in_limiter(x)) ? geqdsk.B(x) : ZERO_V; });

    for (auto &item : particles()) {
        Real ratio = db.getValue("Particles." + item.first + ".ratio", 1.0);
        *item.second->rho = rho0 * ratio;
    }

    TryInitialize();
}
void EMTokamakWorker::Finalize() { base_type::Finalize(); }
void EMTokamakWorker::Finalize(Real data_time) { TryFinalize(); }

void EMTokamakWorker::NextTimeStep(Real data_time, Real dt) {
    PreProcess();
    base_type::NextTimeStep(data_time, dt);
};

void EMTokamakWorker::SetPhysicalBoundaryConditions(Real data_time) {
    base_type::SetPhysicalBoundaryConditions(data_time);
    if (J_src_fun) {
        J1.Assign(model()->select(EDGE, "J_SRC"), [&](point_type const &x) -> Vec3 { return J_src_fun(x, data_time); });
    }
    if (E_src_fun) {
        E.Assign(model()->select(EDGE, "E_SRC"), [&](point_type const &x) -> Vec3 { return E_src_fun(x, data_time); });
    }
};

void EMTokamakWorker::SetPhysicalBoundaryConditionE(Real time) {
    E.Assign(model()->interface(EDGE, "PLASMA", "VACUUM"), 0);
}

void EMTokamakWorker::SetPhysicalBoundaryConditionB(Real time) {
    B.Assign(model()->interface(FACE, "PLASMA", "VACUUM"), 0);
}
}