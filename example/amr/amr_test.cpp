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
//        VERBOSE << "box = " << mesh()->dx() << " time = " << data_time << " dt =" << dt << std::endl;
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
    typedef manifold::CartesianManifold mesh_type; //DummyMesh //
    logger::set_stdout_level(100);
    auto worker = std::make_shared<AMRTest<mesh_type>>();

    worker->print(std::cout);

    auto integrator = simpla::create_time_integrator("AMR_TEST", worker);

    /** test.3d.input */

    /**

    // Refer to geom::CartesianGridGeometry and its base classes for input
    CartesianGeometry{
       domain_boxes	= [ (0,0,0) , (14,9,9) ]

       x_lo = 0.e0 , 0.e0 , 0.e0    // lower end of computational domain.
       x_up = 30.e0 , 20.e0 , 20.e0 // upper end of computational domain.
    }

    // Refer to hier::PatchHierarchy for input
    PatchHierarchy {
       max_levels = 3        // Maximum number of levels in hierarchy.

    // Note: For the following regridding information, data is required for each
    //       potential in the patch hierarchy; i.e., levels 0 thru max_levels-1.
    //       If more data values than needed are given, only the number required
    //       will be read in.  If fewer values are given, an error will result.
    //
    // Specify coarsening ratios for each level 1 through max_levels-1

       ratio_to_coarser {             // vector ratio to next coarser level
          level_1 = 2 , 2 , 2
          level_2 = 2 , 2 , 2
          level_3 = 2 , 2 , 2
       }

       largest_patch_size {
          level_0 = 40 , 40 , 40  // largest patch allowed in hierarchy
          // all finer levels will use same values as level_0...
       }

       smallest_patch_size {
          level_0 = 9 , 9 , 9
          // all finer levels will use same values as level_0...
       }

    }

    // Refer to mesh::GriddingAlgorithm for input
    GriddingAlgorithm{
    }

    // Refer to mesh::BergerRigoutsos for input
    BergerRigoutsos {
       sort_output_nodes = TRUE // Makes results repeatable.
       efficiency_tolerance   = 0.85e0    // min % of tag cells in new patch level
       combine_efficiency     = 0.95e0    // chop box if sum of volumes of smaller
                                          // boxes < efficiency * vol of large box
    }

    // Refer to mesh::StandardTagAndInitialize for input
    StandardTagAndInitialize {
       tagging_method = "GRADIENT_DETECTOR"
    }


    // Refer to algs::HyperbolicLevelIntegrator for input
    HyperbolicLevelIntegrator{
       cfl                       = 0.9e0    // max cfl factor used in problem
       cfl_init                  = 0.9e0    // initial cfl factor
       lag_dt_computation        = TRUE
       use_ghosts_to_compute_dt  = TRUE
    }

    // Refer to algs::TimeRefinementIntegrator for input
    TimeRefinementIntegrator{
       start_time           = 0.e0     // initial simulation time
       end_time             = 100.e0   // final simulation time
       grow_dt              = 1.1e0    // growth factor for timesteps
       max_integrator_steps = 10       // max number of simulation timesteps
    }

    // Refer to mesh::TreeLoadBalancer for input
    LoadBalancer {
       // using default TreeLoadBalancer configuration
    }
     */


    integrator->db["CartesianGeometry"]["domain_boxes_0"] = index_box_type{{0, 0, 0},
                                                                           {NX, NY, NZ}};

    integrator->db["CartesianGeometry"]["periodic_dimension"] = nTuple<int, 3>{1, 1, 1};
    integrator->db["CartesianGeometry"]["x_lo"] = nTuple<double, 3>{0, 0, 0};
    integrator->db["CartesianGeometry"]["x_up"] = nTuple<double, 3>{NX, NY, NZ};

    integrator->db["PatchHierarchy"]["max_levels"] = int(3); // Maximum number of levels in hierarchy.
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_1"] = nTuple<int, 3>{2, 2, 1};
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_2"] = nTuple<int, 3>{2, 2, 1};
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_3"] = nTuple<int, 3>{2, 2, 1};
    integrator->db["PatchHierarchy"]["largest_patch_size"]["level_0"] = nTuple<int, 3>{32, 32, 32};
    integrator->db["PatchHierarchy"]["smallest_patch_size"]["level_0"] = nTuple<int, 3>{4, 4, 4};

    integrator->db["GriddingAlgorithm"];


    integrator->db["BergerRigoutsos"]["sort_output_nodes"] = true;// Makes results repeatable.
    integrator->db["BergerRigoutsos"]["efficiency_tolerance"] = 0.85;  // min % of tag cells in new patch level
    integrator->db["BergerRigoutsos"]["combine_efficiency"] = 0.95;  // chop box if sum of volumes of smaller
//    // boxes < efficiency * vol of large box


    // Refer to mesh::StandardTagAndInitialize for input
    integrator->db["StandardTagAndInitialize"]["tagging_method"] = std::string("GRADIENT_DETECTOR");

    // Refer to algs::HyperbolicLevelIntegrator for input
    integrator->db["HyperbolicLevelIntegrator"]["cfl"] = 0.9;  // max cfl factor used in problem
    integrator->db["HyperbolicLevelIntegrator"]["cfl_init"] = 0.9; // initial cfl factor
    integrator->db["HyperbolicLevelIntegrator"]["lag_dt_computation"] = true;
    integrator->db["HyperbolicLevelIntegrator"]["use_ghosts_to_compute_dt"] = true;

    // Refer to algs::TimeRefinementIntegrator for input
    integrator->db["TimeRefinementIntegrator"]["start_time"] = 0.e0; // initial simulation time
    integrator->db["TimeRefinementIntegrator"]["end_time"] = 20.e0;  // final simulation time
    integrator->db["TimeRefinementIntegrator"]["grow_dt"] = 1.1e0;  // growth factor for timesteps
    integrator->db["TimeRefinementIntegrator"]["max_integrator_steps"] = 5000;  // max number of simulation timesteps

    // Refer to mesh::TreeLoadBalancer for input
    integrator->db["LoadBalancer"];

    integrator->deploy();
    integrator->check_point();


    INFORM << "***********************************************" << std::endl;


    while (integrator->remaining_steps())
    {
        integrator->next_step(0.1);
        integrator->check_point();
    }


    INFORM << "***********************************************" << std::endl;

    integrator->tear_down();

    integrator.reset();

    INFORM << " DONE !" << std::endl;
    DONE;

}

//    index_type lo[3] = {0, 0, 0}, hi[3] = {40, 50, 60};
//    index_type lo1[3] = {10, 20, 30}, hi1[3] = {20, 30, 40};
//
//    size_type gw[3] = {0, 0, 0};
//
//    auto atlas = std::make_shared<mesh::Atlas>();
//    auto m0 = atlas->add<DummyMesh>("FirstLevel", lo, hi, gw, 0);
//    auto m1 = atlas->refine(m0, lo1, hi1);
//
//    std::cout << *atlas << std::endl;
//
//    auto worker = std::make_shared<AMRTest<DummyMesh>>();
//    worker->move_to(m0);
//    TRY_CALL(worker->deploy());
//    worker->move_to(m1);
//    TRY_CALL(worker->deploy());
//    worker->E = 1.2;
//    worker->next_step(1.0);
//    std::cout << " Worker = {" << *worker << "}" << std::endl;
//    std::cout << "E = {" << worker->E << "}" << std::endl;
//    std::cout << "E = {" << *worker->E.attribute() << "}" << std::endl;
//
//    auto m = std::make_shared<mesh::MeshBlock>();
//
//    auto attr = mesh::Attribute::clone();
//    auto f = attr->clone(m);