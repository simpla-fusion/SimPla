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


#include "SAMRAIWorkerHyperbolic.h"

namespace simpla
{


struct SAMRAIWrapperContext : public simulation::ContextBase
{
    SAMRAIWrapperContext()
    {

    }

    ~SAMRAIWrapperContext()
    {

    }

    void setup(int argc, char *argv[]);

    void deploy();

    void teardown();

    std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const { return os; };

    toolbox::IOStream &load(toolbox::IOStream &is) { return is; };

    toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb) {};

    std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const {};

    size_type step() const;

    Real time() const;

    void next_time_step(Real dt);

private:

    boost::shared_ptr<SAMRAIWorkerHyperbolic> patch_worker;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;
//
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


void SAMRAIWrapperContext::setup(int argc, char *argv[])
{

    /*
     * Initialize SAMRAI::tbox::MPI.
     */

    SAMRAI::tbox::SAMRAI_MPI::init(&argc, &argv);
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

    if ((argc != 2) && (argc != 4))
    {
        SAMRAI::tbox::pout << "USAGE:  " << argv[0] << " <input filename> "
                           << "<restart dir> <restore number> [options]\n"
                           << "  options:\n"
                           << "  none at this time"
                           << std::endl;
        SAMRAI::tbox::SAMRAI_MPI::abort();
        return;
    } else
    {
        input_filename = argv[1];
        if (argc == 4)
        {
            restart_read_dirname = argv[2];
            restore_num = atoi(argv[3]);

            is_from_restart = true;
        }
    }

    SAMRAI::tbox::plog << "input_filename = " << input_filename << std::endl;
    SAMRAI::tbox::plog << "restart_read_dirname = " << restart_read_dirname << std::endl;
    SAMRAI::tbox::plog << "restore_num = " << restore_num << std::endl;

    /*
     * Create input database and parse all data in input file.
     */

    boost::shared_ptr<SAMRAI::tbox::InputDatabase> input_db(new SAMRAI::tbox::InputDatabase("input_db"));

    SAMRAI::tbox::InputManager::getManager()->parseInputFile(input_filename, input_db);

    /*
     * Retrieve "GlobalInputs" section of the input database and set
     * values accordingly.
     */

    if (input_db->keyExists("GlobalInputs"))
    {
        boost::shared_ptr<SAMRAI::tbox::Database> global_db(input_db->getDatabase("GlobalInputs"));
#ifdef SGS
        if (global_db->keyExists("tag_clustering_method")) {
       std::string tag_clustering_method = global_db->getString("tag_clustering_method");
       SAMRAI::mesh::BergerRigoutsos::setClusteringOption(tag_clustering_method);
    }
#endif
        if (global_db->keyExists("call_abort_in_serial_instead_of_exit"))
        {
            bool flag = global_db->getBool("call_abort_in_serial_instead_of_exit");
            SAMRAI::tbox::SAMRAI_MPI::setCallAbortInSerialInsteadOfExit(flag);
        }
    }

    /*
     * Retrieve "Main" section of the input database.  First, read
     * dump information, which is used for writing plot files.
     * Second, if proper restart information was given on command
     * line, and the restart interval is non-zero, create a restart
     * database.
     */

    boost::shared_ptr<SAMRAI::tbox::Database> main_db(input_db->getDatabase("Main"));

    const SAMRAI::tbox::Dimension dim(static_cast<unsigned short>(main_db->getInteger("dim")));

    const std::string base_name = main_db->getStringWithDefault("base_name", "unnamed");

    const std::string log_filename = main_db->getStringWithDefault("log_filename", base_name + ".log");

    bool log_all_nodes = false;
    if (main_db->keyExists("log_all_nodes")) { log_all_nodes = main_db->getBool("log_all_nodes"); }
    if (log_all_nodes) { SAMRAI::tbox::PIO::logAllNodes(log_filename); }
    else { SAMRAI::tbox::PIO::logOnlyNodeZero(log_filename); }


    bool use_refined_timestepping = true;

    if (main_db->keyExists("timestepping"))
    {
        std::string timestepping_method = main_db->getString("timestepping");

        if (timestepping_method == "SYNCHRONIZED") { use_refined_timestepping = false; }
    }




    /*
     * Create major algorithm and data objects which comprise application.
     * Each object will be initialized either from input data or restart
     * files, or a combination of both.  Refer to each class constructor
     * for details.  For more information on the composition of objects
     * for this application, see comments at top of file.
     */

    grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(
            dim,
            "CartesianGeometry",
            input_db->getDatabase("CartesianGeometry"));

    auto patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>(
            "PatchHierarchy",
            grid_geometry,
            input_db->getDatabase("PatchHierarchy"));
    //---------------------------------
    patch_worker = boost::make_shared<SAMRAIWorkerHyperbolic>(
            "LinAdv",
            dim,
            input_db->getDatabase("LinAdv"),
            grid_geometry);
    //---------------------------------

    hyp_level_integrator = boost::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
            "HyperbolicLevelIntegrator",
            input_db->getDatabase("HyperbolicLevelIntegrator"),
            patch_worker.get(),
            use_refined_timestepping);

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize",
            hyp_level_integrator.get(),
            input_db->getDatabase("StandardTagAndInitialize"));


    //---------------------------------

    auto box_generator = boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(
            dim,
            input_db->getDatabaseWithDefault("BergerRigoutsos", boost::shared_ptr<SAMRAI::tbox::Database>()));

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(
            dim,
            "LoadBalancer",
            input_db->getDatabase("LoadBalancer"));
    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
            patch_hierarchy,
            "GriddingAlgorithm",
            input_db->getDatabase("GriddingAlgorithm"),
            error_detector,
            box_generator,
            load_balancer);
    //---------------------------------
    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            input_db->getDatabase("TimeRefinementIntegrator"),
            patch_hierarchy,
            hyp_level_integrator,
            gridding_algorithm);

    const std::string viz_dump_dirname = main_db->getStringWithDefault("viz_dump_dirname", base_name + ".visit");
    int visit_number_procs_per_file = 1;

    visit_data_writer = boost::make_shared<appu::VisItDataWriter>(
            dim,
            "LinAdv VisIt Writer",
            viz_dump_dirname,
            visit_number_procs_per_file);
    patch_worker->registerVisItDataWriter(visit_data_writer);
    /*
     * After creating all objects and initializing their state, we
     * print the input database and variable database contents
     * to the log file.
     */

    MESSAGE << "\nCheck input data and variables before simulation:" << std::endl;
    MESSAGE << "Input database..." << std::endl;
    input_db->printClassData(SAMRAI::tbox::plog);
    MESSAGE << "\nVariable database..." << std::endl;
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(SAMRAI::tbox::plog);

    MESSAGE << "\nCheck Linear Advection data... " << std::endl;
    patch_worker->printClassData(SAMRAI::tbox::plog);


}

void SAMRAIWrapperContext::deploy()
{
    time_integrator->initializeHierarchy();
};

void SAMRAIWrapperContext::next_time_step(Real dt) { time_integrator->advanceHierarchy(dt); }

Real SAMRAIWrapperContext::time() const { return static_cast<Real>( time_integrator->getIntegratorTime()); }

size_type SAMRAIWrapperContext::step() const { return static_cast<size_type>( time_integrator->getIntegratorStep()); }

void SAMRAIWrapperContext::teardown()
{


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