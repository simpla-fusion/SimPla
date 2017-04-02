//
// Created by salmon on 16-11-29.
//

#include <simpla/SIMPLA_config.h>

#include <simpla/algebra/all.h>
#include <simpla/engine/Atlas.h>
#include <simpla/engine/Worker.h>
#include <simpla/model/GEqdsk.h>
#include <simpla/physics/Constants.h>
//#include <simpla/predefine/mesh/CartesianGeometry.h>
#include <simpla/predefine/mesh/CylindricalGeometry.h>
#include <iostream>
#include "../em/EMFluid.h"

namespace simpla {
using namespace engine;
// using namespace model;

class EMTokamakWorker;
std::shared_ptr<engine::Mesh> create_mesh() { return std::make_shared<mesh::CylindricalGeometry>(); }

std::shared_ptr<engine::Worker> create_worker() {
    return std::dynamic_pointer_cast<engine::Worker>(std::make_shared<EMTokamakWorker>());
}

class EMTokamakWorker : public EMFluid<mesh::CylindricalGeometry> {
   public:
    SP_OBJECT_HEAD(EMTokamakWorker, EMFluid<mesh::CylindricalGeometry>);
    explicit EMTokamakWorker() : base_type() {}
    ~EMTokamakWorker() {}

    virtual void Initialize();
    virtual void PreProcess();
    virtual void Process(){};
    virtual void PostProcess();
    virtual void Finalize();
    virtual void Initialize(Real data_time);
    virtual void Finalize(Real data_time);
    virtual void NextTimeStep(Real data_time, Real dt);
    virtual void SetPhysicalBoundaryConditions(Real data_time);
    virtual void SetPhysicalBoundaryConditionE(Real time);
    virtual void SetPhysicalBoundaryConditionB(Real time);

    field_type<VERTEX> psi{this, "name"_="psi"};
    std::function<Vec3(point_type const &, Real)> J_src_fun;
    std::function<Vec3(point_type const &, Real)> E_src_fun;
};

void EMTokamakWorker::Initialize() {
    base_type::Initialize();
    // first run, only Load configure, m_mesh_=nullptr

    db()->Get("Particles")
        ->cast_as<DataTable>()
        .Foreach([&](std::string const &key, std::shared_ptr<data::DataEntity> const &item) {
            AddSpecies(key, item->cast_as<DataTable>());
        });

    //    db.SetValue("GetBoundBox", geqdsk.box());
};

void EMTokamakWorker::PreProcess() { base_type::PreProcess(); }
void EMTokamakWorker::PostProcess() { base_type::PostProcess(); }

void EMTokamakWorker::Initialize(Real data_time) {
    PreProcess();
    //    rho0.Assign([&](point_type const &x) -> Real { return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0;
    //    });
    //    psi.Assign([&](point_type const &x) -> Real { return geqdsk.psi(x); });

    nTuple<Real, 3> ZERO_V{0, 0, 0};
    //    B0.Assign([&](point_type const &x) -> Vec3 { return (geqdsk.in_limiter(x)) ? geqdsk.B(x) : ZERO_V; });
    for (auto &item : GetSpecies()) {
        Real ratio = db()->GetValue("Particles." + item.first + ".ratio", 1.0);
        *item.second->rho = rho0 * ratio;
    }
}
void EMTokamakWorker::Finalize() { base_type::Finalize(); }
void EMTokamakWorker::Finalize(Real data_time) {}

void EMTokamakWorker::NextTimeStep(Real data_time, Real dt) {
    PreProcess();
    base_type::NextTimeStep(data_time, dt);
};

void EMTokamakWorker::SetPhysicalBoundaryConditions(Real data_time) {
    base_type::SetPhysicalBoundaryConditions(data_time);
    //    if (J_src_fun) {
    //        J1.Assign(model()->select(EDGE, "J_SRC"), [&](point_type const &x) -> Vec3 { return J_src_fun(x,
    //        data_time); });
    //    }
    //    if (E_src_fun) {
    //        E.Assign(model()->select(EDGE, "E_SRC"), [&](point_type const &x) -> Vec3 { return E_src_fun(x,
    //        data_time); });
    //    }
};

void EMTokamakWorker::SetPhysicalBoundaryConditionE(Real time) {
    //    E.Assign(model()->interface(EDGE, "PLASMA", "VACUUM"), 0);
}

void EMTokamakWorker::SetPhysicalBoundaryConditionB(Real time) {
    //    B.Assign(model()->interface(FACE, "PLASMA", "VACUUM"), 0);
}
}//namespace simpla {
