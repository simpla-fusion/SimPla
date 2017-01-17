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

std::shared_ptr<mesh::Worker> create_worker() {
    return std::dynamic_pointer_cast<mesh::Worker>(std::make_shared<EMTokamakWorker>());
}

class EMTokamakWorker : public EMFluid<mesh::CylindricalGeometry> {
   public:
    SP_OBJECT_HEAD(EMTokamakWorker, EMFluid<mesh::CylindricalGeometry>);

    virtual void deploy();

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time);

    virtual void finalize(Real data_time);

    virtual void next_time_step(Real data_time, Real dt);

    virtual void set_physical_boundary_conditions(Real data_time);

    virtual void set_physical_boundary_conditions_E(Real time);

    virtual void set_physical_boundary_conditions_B(Real time);

    GEqdsk geqdsk;

    field_type<VERTEX> psi{this, {"name"_ = "psi"}};

    std::function<Vec3(point_type const &, Real)> J_src_fun;

    std::function<Vec3(point_type const &, Real)> E_src_fun;
};

void EMTokamakWorker::deploy() {
    base_type::deploy();

    chart(std::make_shared<mesh_type>());
    // first run, only load configure, m_chart_=nullptr
    geqdsk.load(db.get_value("GEqdsk", "geqdsk.gfile"));

    db.as_table("Particles").foreach ([&](std::string const &key, data::DataEntity const &item) {
        add_particle(key, item.as_table());
    });

    db.set_value("bound_box", geqdsk.box());

    //    model()->add_object("VACUUM", geqdsk.limiter_gobj());
    //    model()->add_object("PLASMA", geqdsk.boundary_gobj());
};

void EMTokamakWorker::pre_process() {
    if (is_valid()) {
        return;
    } else {
        base_type::pre_process();
    }
}

void EMTokamakWorker::post_process() {
    if (!is_valid()) {
        return;
    } else {
        base_type::post_process();
    }
}

void EMTokamakWorker::initialize(Real data_time) {
    pre_process();

    rho0.assign([&](point_type const &x) { return (geqdsk.in_boundary(x)) ? geqdsk.profile("ne", x) : 0.0; });

    psi.assign([&](point_type const &x) { return geqdsk.psi(x); });

    nTuple<Real, 3> ZERO_V{0, 0, 0};

    B0.assign([&](point_type const &x) { return (geqdsk.in_limiter(x)) ? geqdsk.B(x) : ZERO_V; });

    for (auto &item : particles()) {
        Real ratio = db.get_value("Particles." + item.first + ".ratio", 1.0);
        *item.second->rho = rho0 * ratio;
    }

    base_type::initialize(data_time);
}

void EMTokamakWorker::finalize(Real data_time) {
    post_process();
    base_type::finalize(data_time);
}

void EMTokamakWorker::next_time_step(Real data_time, Real dt) {
    pre_process();
    base_type::next_time_step(data_time, dt);
};

void EMTokamakWorker::set_physical_boundary_conditions(Real data_time) {
    base_type::set_physical_boundary_conditions(data_time);
    if (J_src_fun) {
        J1.assign(model()->select(EDGE, "J_SRC"), [&](point_type const &x) { return J_src_fun(x, data_time); });
    }
    if (E_src_fun) {
        E.assign(model()->select(EDGE, "E_SRC"), [&](point_type const &x) { return E_src_fun(x, data_time); });
    }
};

void EMTokamakWorker::set_physical_boundary_conditions_E(Real time) {
    E.assign(model()->interface(EDGE, "PLASMA", "VACUUM"), 0);
}

void EMTokamakWorker::set_physical_boundary_conditions_B(Real time) {
    B.assign(model()->interface(FACE, "PLASMA", "VACUUM"), 0);
}
}