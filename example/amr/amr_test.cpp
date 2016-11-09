//
// Created by salmon on 16-11-4.
//

#include <simpla/SIMPLA_config.h>

#include <iostream>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Atlas.h>
#include <simpla/physics/Field.h>
#include <simpla/manifold/Calculus.h>
#include "SAMRAITimeIntegrator.h"

using namespace simpla;
//
//class DummyMesh : public mesh::MeshBlock
//{
//public:
//    static constexpr unsigned int ndims = 3;
//
//    SP_OBJECT_HEAD(DummyMesh, mesh::MeshBlock)
//
//    template<typename ...Args>
//    DummyMesh(Args &&...args):mesh::MeshBlock(std::forward<Args>(args)...) {}
//
//    ~DummyMesh() {}
//
//    template<typename TV, mesh::MeshEntityType IFORM> using data_block_type= mesh::DataBlockArray<Real, IFORM>;
//
//    virtual std::shared_ptr<mesh::MeshBlock> clone() const
//    {
//        return std::dynamic_pointer_cast<mesh::MeshBlock>(std::make_shared<DummyMesh>());
//    };
//
//    template<typename ...Args>
//    Real eval(Args &&...args) const { return 1.0; };
//};
//
//template<typename TM>
//struct AMRTest : public mesh::Worker
//{
//    typedef TM mesh_type;
//
//    SP_OBJECT_HEAD(AMRTest, mesh::Worker);
//
//    template<typename TV, mesh::MeshEntityType IFORM> using field_type=Field<TV, mesh_type, index_const<IFORM>>;
//    field_type<Real, mesh::VERTEX> phi{"phi", this};
//    field_type<Real, mesh::EDGE> E{"E", this};
//    field_type<Real, mesh::FACE> B{"B", this};
//
//    void next_time_step(Real dt)
//    {
//        E = grad(-2.0 * phi) * dt;
//        phi -= diverge(E) * 3.0 * dt;
//    }
//
//};

int main(int argc, char **argv)
{
    auto integrator = simpla::create_samrai_time_integrator("samrai_integrator");

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


    integrator->db["CartesianGeometry"]["domain_boxes_0"] = index_box_type{{0,  0,  0},
                                                                           {15, 15, 15}};

    integrator->db["CartesianGeometry"]["x_lo"] = nTuple<double, 3>{0, 0, 0};
    integrator->db["CartesianGeometry"]["x_up"] = nTuple<double, 3>{1, 1, 1};
    integrator->db["PatchHierarchy"]["max_levels"] = int(3); // Maximum number of levels in hierarchy.
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_1"] = nTuple<int, 3>{2, 2, 2};
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_2"] = nTuple<int, 3>{2, 2, 2};
    integrator->db["PatchHierarchy"]["ratio_to_coarser"]["level_3"] = nTuple<int, 3>{2, 2, 2};
    integrator->db["PatchHierarchy"]["largest_patch_size"]["level_0"] = nTuple<int, 3>{40, 40, 40};
    integrator->db["PatchHierarchy"]["smallest_patch_size"]["level_0"] = nTuple<int, 3>{9, 9, 9};
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
    integrator->db["TimeRefinementIntegrator"]["end_time"] = 10.e0;  // final simulation time
    integrator->db["TimeRefinementIntegrator"]["grow_dt"] = 1.1e0;  // growth factor for timesteps
    integrator->db["TimeRefinementIntegrator"]["max_integrator_steps"] = 10;  // max number of simulation timesteps

    // Refer to mesh::TreeLoadBalancer for input
    integrator->db["LoadBalancer"];

    integrator->deploy();
    integrator->print(std::cout);
    integrator->tear_down();
    integrator.reset();

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
//    worker->next_time_step(1.0);
//    std::cout << " Worker = {" << *worker << "}" << std::endl;
//    std::cout << "E = {" << worker->E << "}" << std::endl;
//    std::cout << "E = {" << *worker->E.attribute() << "}" << std::endl;
//
//    auto m = std::make_shared<mesh::MeshBlock>();
//
//    auto attr = mesh::Attribute::clone();
//    auto f = attr->clone(m);