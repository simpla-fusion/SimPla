//
// Created by salmon on 16-10-24.
//

// Headers for SimPla
#include <simpla/SIMPLA_config.h>

#include <memory>
#include <string>
#include <cmath>

#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Patch.h>
#include <simpla/simulation/Context.h>
#include <simpla/simulation/Worker.h>

// Headers for SAMRAI
#include <SAMRAI/SAMRAI_config.h>

#include <SAMRAI/algs/HyperbolicLevelIntegrator.h>
#include <SAMRAI/algs/TimeRefinementIntegrator.h>
#include <SAMRAI/algs/TimeRefinementLevelStrategy.h>

#include <SAMRAI/mesh/BergerRigoutsos.h>
#include <SAMRAI/mesh/GriddingAlgorithm.h>
#include <SAMRAI/mesh/CascadePartitioner.h>
#include <SAMRAI/mesh/StandardTagAndInitialize.h>
#include <SAMRAI/mesh/CascadePartitioner.h>

#include <SAMRAI/hier/Index.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/hier/BoundaryBox.h>
#include <SAMRAI/hier/BoxContainer.h>
#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/hier/PatchDataRestartManager.h>
#include <SAMRAI/hier/VariableDatabase.h>

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/geom/CartesianPatchGeometry.h>

#include <SAMRAI/pdat/CellData.h>
#include <SAMRAI/pdat/CellIndex.h>
#include <SAMRAI/pdat/CellIterator.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/FaceIndex.h>
#include <SAMRAI/pdat/FaceVariable.h>

#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/Utilities.h>
#include <SAMRAI/tbox/MathUtilities.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/BalancedDepthFirstTree.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/Utilities.h>

#include <SAMRAI/appu/VisItDataWriter.h>
#include <SAMRAI/appu/BoundaryUtilityStrategy.h>
#include <SAMRAI/appu/CartesianBoundaryDefines.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities2.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities3.h>
#include <boost/shared_ptr.hpp>
#include <simpla/toolbox/DataBase.h>


#include "SAMRAIWorkerHyperbolic.h"

namespace simpla
{


struct SAMRAIWrapperContext : public simulation::ContextBase
{
    SAMRAIWrapperContext() {}

    ~SAMRAIWrapperContext() {}

    void initialize(int argc = 0, char **argv = nullptr);

    void load(std::shared_ptr<toolbox::DataBase> const &);

    void save(toolbox::DataBase *);

    void deploy();

    bool is_valid() const;

    void teardown();

    std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    void registerAttribute(std::string const &, std::shared_ptr<mesh::AttributeBase> &, int flag = 0);

    toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    size_type step() const;

    Real time() const;

    void next_time_step(Real dt);

private:
    bool m_is_valid_ = false;
    boost::shared_ptr<SAMRAIWorkerHyperbolic> patch_worker;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;

    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

//    boost::shared_ptr<SAMRAI::mesh::StandardTagAndInitialize> error_detector;
//
//    boost::shared_ptr<SAMRAI::mesh::BergerRigoutsos> box_generator;
//
//    boost::shared_ptr<SAMRAI::mesh::CascadePartitioner> load_balancer;
//
//    boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithm> gridding_algorithm;

    boost::shared_ptr<SAMRAI::algs::TimeRefinementIntegrator> time_integrator;

    // VisItDataWriter is only present if HDF is available
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> visit_data_writer;


    bool write_restart = false;
    int restart_interval = 0;

    std::string restart_write_dirname;

    bool viz_dump_data = false;
    int viz_dump_interval = 1;

};

void SAMRAIWrapperContext::initialize(int argc, char **argv)
{
    /*
     * Initialize SAMRAI::tbox::MPI.
     */
    tbox::SAMRAI_MPI::init(&argc, &argv);

    SAMRAI::tbox::SAMRAIManager::initialize();
    /*
     * Initialize SAMRAI, enable logging, and process command line.
     */
    SAMRAI::tbox::SAMRAIManager::startup();
    const SAMRAI::tbox::SAMRAI_MPI &mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());


    std::string input_filename;
    std::string restart_read_dirname;
    int restore_num = 0;

    bool is_from_restart = false;
}


void SAMRAIWrapperContext::load(std::shared_ptr<toolbox::DataBase> const &db)
{
    m_is_valid_ = false;

    const SAMRAI::tbox::Dimension dim(3);
    bool use_refined_timestepping = true;
    /*

     * Create major algorithm and data objects which comprise application.
     * Each object will be initialized either from input data or restart
     * files, or a combination of both.  Refer to each class constructor
     * for details.  For more information on the composition of objects
     * for this application, see comments at top of file.
     */


    //---------------------------------


    auto CartesianGeometry_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("CartesianGeometry");


    size_t ndims = 3;
    int i_lo[ndims] = {0, 0, 0}, i_up[ndims] = {40, 40, 40};
    double x_lo[ndims] = {0, 0, 0}, x_up[ndims] = {1, 1, 1};
    int periodic_dimension[ndims];
    std::vector<SAMRAI::tbox::DatabaseBox> box_vec;
    box_vec.emplace_back(SAMRAI::tbox::DatabaseBox(dim, i_lo, i_up));
    CartesianGeometry_db->putDatabaseBoxVector("domain_boxes", box_vec);
    CartesianGeometry_db->putDoubleArray("x_lo", x_lo, ndims);
    CartesianGeometry_db->putDoubleArray("x_up", x_up, ndims);
    CartesianGeometry_db->putIntegerArray("periodic_dimension", periodic_dimension, ndims);


    grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(dim, "CartesianGeometry",
                                                                            CartesianGeometry_db);
    grid_geometry->printClassData(std::cout);
    //---------------------------------

    auto PatchHierarchy_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("PatchHierarchy");


    {
        auto ratio_to_coarser = PatchHierarchy_db->putDatabase("ratio_to_coarser");
        int ratio_to_coarser_d[4][3] = {
                {2, 2, 2},
                {2, 2, 2},
                {2, 2, 2},
                {2, 2, 2}
        };
        ratio_to_coarser->putIntegerArray("level_1", ratio_to_coarser_d[0], ndims);
        ratio_to_coarser->putIntegerArray("level_2", ratio_to_coarser_d[1], ndims);
        ratio_to_coarser->putIntegerArray("level_3", ratio_to_coarser_d[2], ndims);

        auto smallest_patch_size = PatchHierarchy_db->putDatabase("smallest_patch_size");

        int smallest_patch_size_d[3] = {16, 16, 16};

        smallest_patch_size->putIntegerArray("level_0", smallest_patch_size_d, ndims);

        int largest_patch_size_d[3] = {40, 40, 40};

        auto largest_patch_size = PatchHierarchy_db->putDatabase("auto");

        largest_patch_size->putIntegerArray("level_0", largest_patch_size_d, ndims);

        PatchHierarchy_db->putInteger("proper_nesting_buffer", 1);
        PatchHierarchy_db->putInteger("max_levels", 4);
        PatchHierarchy_db->putBool("allow_patches_smaller_than_ghostwidth", false);
        PatchHierarchy_db->putBool("allow_patches_smaller_than_minimum_size_to_prevent_overlap", false);
    }
    auto patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>("PatchHierarchy", grid_geometry,
                                                                            PatchHierarchy_db);

    patch_hierarchy->recursivePrint(std::cout, "", 1);

    //---------------------------------
    patch_worker = boost::make_shared<SAMRAIWorkerHyperbolic>("unnamed", dim, grid_geometry);
    //---------------------------------
    auto HyperbolicLevelIntegrator_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("HyperbolicLevelIntegrator");

    HyperbolicLevelIntegrator_db->putDouble("cfl", 0.5);
    HyperbolicLevelIntegrator_db->putDouble("cfl_init", 0.5);
    HyperbolicLevelIntegrator_db->putBool("lag_dt_computation", true);
    HyperbolicLevelIntegrator_db->putBool("use_ghosts_to_compute_dt", false);


    hyp_level_integrator = boost::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
            "HyperbolicLevelIntegrator", HyperbolicLevelIntegrator_db, patch_worker.get(), use_refined_timestepping);

    hyp_level_integrator->printClassData(std::cout);
    //---------------------------------
    auto StandardTagAndInitialize_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("StandardTagAndInitialize");

    StandardTagAndInitialize_db->putString("tagging_method", "GRADIENT_DETECTOR");

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize", hyp_level_integrator.get(), StandardTagAndInitialize_db);

    //---------------------------------
    auto BergerRigoutsos_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("BergerRigoutsos");

    BergerRigoutsos_db->putBool("sort_output_nodes", true);
    BergerRigoutsos_db->putDouble("efficiency_tolerance", 0.85e0);
    BergerRigoutsos_db->putDouble("combine_efficiency", 0.95e0);

    auto box_generator = boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(
            dim, BergerRigoutsos_db);

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());


    //---------------------------------
    auto LoadBalancer_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("LoadBalancer");

    auto load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(
            dim, "LoadBalancer", LoadBalancer_db);

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    load_balancer->printStatistics(std::cout);

    //---------------------------------
    auto GriddingAlgorithm_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("GriddingAlgorithm");

    auto gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
            patch_hierarchy,
            "GriddingAlgorithm",
            GriddingAlgorithm_db,
            error_detector,
            box_generator,
            load_balancer);

    gridding_algorithm->printClassData(std::cout);
    //---------------------------------
    auto TimeRefinementIntegrator_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("TimeRefinementIntegrator");
    TimeRefinementIntegrator_db->putDouble("start_time", 0e0);
    TimeRefinementIntegrator_db->putDouble("end_time", 100.e0);
    TimeRefinementIntegrator_db->putDouble("grow_dt", 1.1e0);
    TimeRefinementIntegrator_db->putDouble("max_integrator_steps", 10);


    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            TimeRefinementIntegrator_db,
            patch_hierarchy,
            hyp_level_integrator,
            gridding_algorithm);

    time_integrator->printClassData(std::cout);

    const std::string viz_dump_dirname("untitled.visit");

    int visit_number_procs_per_file = 1;

    visit_data_writer = boost::make_shared<appu::VisItDataWriter>(
            dim,
            "LinAdv VisIt Writer",
            viz_dump_dirname,
            visit_number_procs_per_file);
    patch_worker->registerVisItDataWriter(visit_data_writer);


}

void SAMRAIWrapperContext::save(toolbox::DataBase *)
{
    assert(is_valid());
    UNIMPLEMENTED;
}


void SAMRAIWrapperContext::registerAttribute(std::string const &, std::shared_ptr<mesh::AttributeBase> &, int flag)
{
    m_is_valid_ = false;

}


void SAMRAIWrapperContext::deploy()
{
    m_is_valid_ = true;
    time_integrator->initializeHierarchy();
};

bool SAMRAIWrapperContext::is_valid() const { return m_is_valid_; }

void SAMRAIWrapperContext::next_time_step(Real dt)
{
    assert(is_valid());
    time_integrator->advanceHierarchy(dt);
}

Real SAMRAIWrapperContext::time() const { return static_cast<Real>( time_integrator->getIntegratorTime()); }

size_type SAMRAIWrapperContext::step() const { return static_cast<size_type>( time_integrator->getIntegratorStep()); }

void SAMRAIWrapperContext::teardown()
{
    m_is_valid_ = false;

    /*
     * At conclusion of simulation, deallocate objects.
     */

    visit_data_writer.reset();

    time_integrator.reset();
//    gridding_algorithm.reset();
//    load_balancer.reset();
//    box_generator.reset();
//    error_detector.reset();
    hyp_level_integrator.reset();
    patch_worker.reset();

//    patch_hierarchy.reset();
//    grid_geometry.reset();

//    input_db.reset();
//    main_db.reset();



    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

}


std::shared_ptr<simulation::ContextBase> create_context(std::string const &name)
{
    return std::make_shared<SAMRAIWrapperContext>();
}
}//namespace simpla

//namespace simpla
//{
//
//namespace detail
//{
//
//template<typename V, mesh::MeshEntityType IFORM> struct SAMRAITraitsPatch;
//template<typename V> struct SAMRAITraitsPatch<V, mesh::VERTEX> { typedef SAMRAI::pdat::NodeData<V> type; };
//template<typename V> struct SAMRAITraitsPatch<V, mesh::EDGE> { typedef SAMRAI::pdat::EdgeData<V> type; };
//template<typename V> struct SAMRAITraitsPatch<V, mesh::FACE> { typedef SAMRAI::pdat::FaceData<V> type; };
//template<typename V> struct SAMRAITraitsPatch<V, mesh::VOLUME> { typedef SAMRAI::pdat::CellData<V> type; };
//
//
//template<typename V, typename M, mesh::MeshEntityType IFORM>
//class SAMRAIWrapperPatch
//        : public SAMRAITraitsPatch<V, IFORM>::type,
//          public mesh::Patch<V, M, IFORM>
//{
//    typedef typename SAMRAITraitsPatch<V, IFORM>::type samari_base_type;
//    typedef mesh::Patch<V, M, IFORM> simpla_base_type;
//public:
//    SAMRAIWrapperPatch(std::shared_ptr<M> const &m, size_tuple const &gw)
//            : samari_base_type(SAMRAI::hier::Box(samraiIndexConvert(std::get<0>(m->index_box())),
//                                                 samraiIndexConvert(std::get<1>(m->index_box())),
//                                                 SAMRAI::hier::BlockId(0)),
//                               1, samraiIntVectorConvert(gw)),
//              simpla_base_type(m->get()) {}
//
//    ~SAMRAIWrapperPatch() {}
//};
//
//
//template<typename V, mesh::MeshEntityType IFORM> struct SAMRAITraitsVariable;
//template<typename V> struct SAMRAITraitsVariable<V, mesh::VERTEX> { typedef SAMRAI::pdat::NodeVariable<V> type; };
//template<typename V> struct SAMRAITraitsVariable<V, mesh::EDGE> { typedef SAMRAI::pdat::EdgeVariable<V> type; };
//template<typename V> struct SAMRAITraitsVariable<V, mesh::FACE> { typedef SAMRAI::pdat::FaceVariable<V> type; };
//template<typename V> struct SAMRAITraitsVariable<V, mesh::VOLUME> { typedef SAMRAI::pdat::CellVariable<V> type; };
//
//template<typename V, typename M, mesh::MeshEntityType IFORM>
//class SAMRAIWrapperAttribute
//        : public SAMRAITraitsVariable<V, IFORM>::type,
//          public mesh::Attribute<SAMRAIWrapperPatch<V, M, IFORM> >
//{
//    typedef typename SAMRAITraitsVariable<V, IFORM>::type samrai_base_type;
//    typedef mesh::Attribute<SAMRAIWrapperPatch<V, M, IFORM>> simpla_base_type;
//public:
//    template<typename TM>
//    SAMRAIWrapperAttribute(std::shared_ptr<TM> const &m, std::string const &name) :
//            samrai_base_type(SAMRAI::tbox::Dimension(M::ndims), name, 1), simpla_base_type(m) {}
//
//    ~SAMRAIWrapperAttribute() {}
//};
//
//
//class SAMRAIWrapperAtlas
//        : public mesh::Atlas,
//          public SAMRAI::hier::PatchHierarchy
//{
//
//    typedef mesh::Atlas simpla_base_type;
//    typedef SAMRAI::hier::PatchHierarchy samrai_base_type;
//public:
//    SAMRAIWrapperAtlas(std::string const &name)
//            : samrai_base_type(name,
//                               boost::shared_ptr<SAMRAI::hier::BaseGridGeometry>(
//                                       new SAMRAI::geom::CartesianGridGeometry(
//                                               SAMRAI::tbox::Dimension(3),
//                                               "CartesianGridGeometry",
//                                               boost::shared_ptr<SAMRAI::tbox::Database>(nullptr)))
//    )
//    {
//
//    }
//
//    ~SAMRAIWrapperAtlas() {}
//};
//
//std::shared_ptr<mesh::AttributeBase>
//create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                      std::shared_ptr<mesh::Atlas> const &m, std::string const &name)
//{
//}
//
//
//std::shared_ptr<mesh::AttributeBase>
//create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                      std::shared_ptr<mesh::MeshBase> const &m, std::string const &name)
//{
//}
//
//std::shared_ptr<mesh::PatchBase>
//create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                  std::shared_ptr<mesh::MeshBase> const &m)
//{
//
//}
//}//namespace detail
//} //namespace simpla