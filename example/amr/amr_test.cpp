//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>
#include <simpla/manifold/pre_define/PreDefine.h>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/physics/Field.h>
#include <simpla/physics/Constants.h>

#include <simpla/manifold/Calculus.h>
#include <simpla/simulation/TimeIntegrator.h>

#define NX 64
#define NY 64
#define NZ 64
#define omega 1.0
using namespace simpla;

class DummyMesh : public mesh::MeshBlock
{
public:
    static constexpr unsigned int ndims = 3;

    SP_OBJECT_HEAD(DummyMesh, mesh::MeshBlock)

    template<typename ...Args>
    DummyMesh(Args &&...args):mesh::MeshBlock(std::forward<Args>(args)...) {}

    ~DummyMesh() {}

    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;

    virtual std::shared_ptr<mesh::MeshBlock> clone() const
    {
        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
    };

    template<typename TV, mesh::MeshEntityType IFORM>
    std::shared_ptr<mesh::DataBlock> create_data_block(void *p) const
    {
        auto b = outer_index_box();

        index_type lo[4] = {std::get<0>(b)[0], std::get<0>(b)[1], std::get<0>(b)[2], 0};
        index_type hi[4] = {std::get<1>(b)[0], std::get<1>(b)[1], std::get<0>(b)[2], 3};
        return std::dynamic_pointer_cast<mesh::DataBlock>(
                std::make_shared<data_block_type<TV, IFORM>>(
                        static_cast<TV *>(p),
                        (IFORM == mesh::VERTEX || IFORM == mesh::VOLUME) ? 3 : 4,
                        lo, hi));
    };


    template<typename ...Args>
    Real eval(Args &&...args) const { return 1.0; };
};

template<typename TM>
struct AMRTest : public mesh::Worker
{
    typedef TM mesh_type;

    AMRTest() : mesh::Worker() {}

    ~AMRTest() {}


    SP_OBJECT_HEAD(AMRTest, mesh::Worker);
    Real m_k_[3] = {TWOPI / NX, TWOPI / NY, TWOPI / NZ};
    template<typename TV, mesh::MeshEntityType IFORM> using field_type=Field<TV, mesh_type, index_const<IFORM>>;
//    field_type<Real, mesh::VERTEX> phi{"phi", this};

    Real epsilon = 1.0;
    Real mu = 1.0;
    field_type<Real, mesh::FACE> B{"B", this};
    field_type<Real, mesh::EDGE> E{"E", this};
    field_type<Real, mesh::EDGE> J{"J", this};
//    field_type<Real, mesh::EDGE> D{"D", this};
//    field_type<Real, mesh::FACE> H{"H", this};


//    field_type<nTuple<Real, 3>, mesh::VERTEX> Ev{"Ev", this};
//    field_type<nTuple<Real, 3>, mesh::VERTEX> Bv{"Bv", this};
    virtual std::shared_ptr<mesh::MeshBlock>
    create_mesh_block(index_type const *lo, index_type const *hi, Real const *dx,
                      Real const *xlo = nullptr, Real const *xhi = nullptr) const
    {
        auto res = std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<mesh_type>(3, lo, hi, dx, xlo, xhi));
        res->deploy();
        return res;
    };


    void initialize(Real data_time)
    {
        E.clear();
        B.clear();
        J.clear();
//        E.foreach([&](point_type const &x)
//                  {
//                      return nTuple<Real, 3>{
//                              std::sin(x[0] * m_k_[0]) * std::sin(x[1] * m_k_[1]) * std::sin(x[2] * m_k_[2]),
//                              0,//  std::cos(x[0] * m_k_[0]) * std::cos(x[1] * m_k_[1]) * std::cos(x[2] * m_k_[2]),
//                              0//  std::sin(x[0] * m_k_[0]) * std::cos(x[1] * m_k_[1]) * std::sin(x[2] * m_k_[2])
//                      };
//                  });

    }

    virtual void setPhysicalBoundaryConditions(double time)
    {

        auto b = mesh()->inner_index_box();

        index_tuple p = {NX / 2, NY / 2, NZ / 2};

        if (toolbox::is_inside(p, b))
        {
            E(p[0], p[1], p[2], 0) = std::sin(omega * time);
        }

    };


    virtual void next_time_step(Real data_time, Real dt)
    {
        VERBOSE << "box = " << mesh()->dx() << " time = " << data_time << " dt =" << dt << std::endl;
        E = E + (curl(B) / mu - J) * dt / epsilon;
        B = B - curl(E) * dt;
    }


};
namespace simpla
{
std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w);
}//namespace simpla

int main(int argc, char **argv)
{
    logger::set_stdout_level(100);

    auto integrator = simpla::create_time_integrator("AMR_TEST",
                                                     std::make_shared<AMRTest<manifold::CartesianManifold>>());

    integrator->deploy();

    integrator->check_point();

    INFORM << "***********************************************" << std::endl;

    while (integrator->remaining_steps())
    {
        integrator->next_step(1);
        integrator->check_point();
    }

    INFORM << "***********************************************" << std::endl;

    integrator->tear_down();

    integrator.reset();

    INFORM << " DONE !" << std::endl;
    DONE;

}

