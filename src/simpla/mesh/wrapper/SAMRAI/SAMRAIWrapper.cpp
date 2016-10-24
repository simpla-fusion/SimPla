//
// Created by salmon on 16-10-24.
//
#include <simpla/toolbox/nTuple.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Patch.h>
#include <simpla/simulation/Context.h>


#include <SAMRAI/SAMRAI_config.h>
#include <SAMRAI/hier/VariableDatabase.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/hier/BaseGridGeometry.h>
#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/pdat/CellData.h>
#include <SAMRAI/pdat/EdgeData.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/NodeData.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/NodeVariable.h>


#include <SAMRAI/appu/VisItDataWriter.h>
#include <SAMRAI/mesh/BergerRigoutsos.h>
#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/mesh/GriddingAlgorithm.h>
#include <SAMRAI/algs/HyperbolicLevelIntegrator.h>
#include <SAMRAI/mesh/CascadePartitioner.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/mesh/StandardTagAndInitialize.h>
#include <SAMRAI/algs/TimeRefinementIntegrator.h>
#include <SAMRAI/algs/TimeRefinementLevelStrategy.h>

// Headers for basic SAMRAI objects

#include <SAMRAI/hier/VariableDatabase.h>
#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/BalancedDepthFirstTree.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/Utilities.h>


#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>


#include "LinAdv.h"

namespace simpla
{
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


struct SAMRAIWrapperContext : public simulation::ContextBase
{
    SAMRAIWrapperContext()
    {

    }

    ~SAMRAIWrapperContext()
    {

    }

    void setup(int argc, char *argv[]);

    void teardown();

    std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

    toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const { return os; };

    toolbox::IOStream &load(toolbox::IOStream &is) { return is; };

    toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    std::shared_ptr<mesh::DomainBase> add_domain(std::shared_ptr<mesh::DomainBase> pb) {};

    std::shared_ptr<mesh::DomainBase> get_domain(uuid id) const {};

    void sync(int level = 0, int flag = 0) {}

    void run(Real dt, int level = 0);

    Real time() const {};

    void time(Real t) {};

    void next_time_step(Real dt) {};


private:

    boost::shared_ptr<LinAdv> linear_advection_model;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;

    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

    boost::shared_ptr<SAMRAI::mesh::StandardTagAndInitialize> error_detector;

    boost::shared_ptr<SAMRAI::mesh::BergerRigoutsos> box_generator;

    boost::shared_ptr<SAMRAI::mesh::CascadePartitioner> load_balancer;

    boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithm> gridding_algorithm;

    boost::shared_ptr<SAMRAI::algs::TimeRefinementIntegrator> time_integrator;

    // VisItDataWriter is only present if HDF is available
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> visit_data_writer;
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
        MESSAGE << "USAGE:  " << argv[0] << " <input filename> "
                << "<restart dir> <restore number> [options]\n"
                << "  options:\n"
                << "  none at this time"
                << std::endl;
        SAMRAI::tbox::SAMRAI_MPI::abort();
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

    LOGGER << "input_filename = " << input_filename << std::endl;
    LOGGER << "restart_read_dirname = " << restart_read_dirname << std::endl;
    LOGGER << "restore_num = " << restore_num << std::endl;

    /*
     * Create input database and parse all data in input file.
     */

    auto input_db = boost::make_shared<SAMRAI::tbox::InputDatabase>("input_db");
    SAMRAI::tbox::InputManager::getManager()->parseInputFile(input_filename, input_db);

    /*
     * Retrieve "GlobalInputs" section of the input database and set
     * values accordingly.
     */

    if (input_db->keyExists("GlobalInputs"))
    {
        boost::shared_ptr<SAMRAI::tbox::Database> global_db(
                input_db->getDatabase("GlobalInputs"));
#ifdef SGS
        if (global_db->keyExists("tag_clustering_method")) {
       std::string tag_clustering_method =
          global_db->getString("tag_clustering_method");
       SAMRAI::mesh::BergerRigoutsos::setClusteringOption(tag_clustering_method);
    }
#endif
        if (global_db->keyExists("call_abort_in_serial_instead_of_exit"))
        {
            bool flag = global_db->
                    getBool("call_abort_in_serial_instead_of_exit");
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

    auto main_db = input_db->getDatabase("Main");

    const SAMRAI::tbox::Dimension dim(static_cast<unsigned short>(main_db->getInteger("dim")));

    const std::string base_name = main_db->getStringWithDefault("base_name", "unnamed");

    const std::string log_filename = main_db->getStringWithDefault("log_filename", base_name + ".log");

    bool log_all_nodes = false;
    if (main_db->keyExists("log_all_nodes"))
    {
        log_all_nodes = main_db->getBool("log_all_nodes");
    }
    if (log_all_nodes)
    {
        SAMRAI::tbox::PIO::logAllNodes(log_filename);
    } else
    {
        SAMRAI::tbox::PIO::logOnlyNodeZero(log_filename);
    }

    int viz_dump_interval = 0;
    if (main_db->keyExists("viz_dump_interval"))
    {
        viz_dump_interval = main_db->getInteger("viz_dump_interval");
    }

    const std::string viz_dump_dirname =
            main_db->getStringWithDefault("viz_dump_dirname", base_name + ".visit");
    int visit_number_procs_per_file = 1;

    const bool viz_dump_data = (viz_dump_interval > 0);

    int restart_interval = 0;
    if (main_db->keyExists("restart_interval"))
    {
        restart_interval = main_db->getInteger("restart_interval");
    }

    const std::string restart_write_dirname = main_db->getStringWithDefault("restart_write_dirname",
                                                                            base_name + ".restart");

    bool use_refined_timestepping = true;
    if (main_db->keyExists("timestepping"))
    {
        std::string timestepping_method = main_db->getString("timestepping");
        if (timestepping_method == "SYNCHRONIZED")
        {
            use_refined_timestepping = false;
        }
    }


    const bool write_restart = (restart_interval > 0) && !(restart_write_dirname.empty());

    /*
     * Get the restart manager and root restart database.  If run is from
     * restart, open the restart file.
     */

    SAMRAI::tbox::RestartManager *restart_manager = SAMRAI::tbox::RestartManager::getManager();

    if (is_from_restart)
    {
        restart_manager->openRestartFile(restart_read_dirname, restore_num, mpi.getSize());
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

    patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>(
            "PatchHierarchy",
            grid_geometry,
            input_db->getDatabase("PatchHierarchy"));

    linear_advection_model = boost::make_shared<LinAdv>(
            "LinAdv",
            dim,
            input_db->getDatabase("LinAdv"),
            grid_geometry);

    hyp_level_integrator = boost::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
            "HyperbolicLevelIntegrator",
            input_db->getDatabase("HyperbolicLevelIntegrator"),
            linear_advection_model.get(),
            use_refined_timestepping);

    error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize",
            hyp_level_integrator.get(),
            input_db->getDatabase("StandardTagAndInitialize"));

    box_generator = boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(
            dim,
            input_db->getDatabaseWithDefault("BergerRigoutsos", boost::shared_ptr<SAMRAI::tbox::Database>()));

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(
            dim,
            "LoadBalancer",
            input_db->getDatabase("LoadBalancer"));

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
            patch_hierarchy,
            "GriddingAlgorithm",
            input_db->getDatabase("GriddingAlgorithm"),
            error_detector,
            box_generator,
            load_balancer);

    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            input_db->getDatabase("TimeRefinementIntegrator"),
            patch_hierarchy,
            hyp_level_integrator,
            gridding_algorithm);

    // VisItDataWriter is only present if HDF is available
    visit_data_writer = boost::make_shared<SAMRAI::appu::VisItDataWriter>(
            dim,
            "LinAdv VisIt Writer",
            viz_dump_dirname,
            visit_number_procs_per_file);

    linear_advection_model->registerVisItDataWriter(visit_data_writer);

    /*
     * Initialize hierarchy configuration and data on all patches.
     * Then, close restart file and write initial state for visualization.
     */

    double dt_now = time_integrator->initializeHierarchy();

    SAMRAI::tbox::RestartManager::getManager()->closeRestartFile();

    /*
     * After creating all objects and initializing their state, we
     * print the input database and variable database contents
     * to the log file.
     */

    LOGGER << "\nCheck input data and variables before simulation:" << std::endl;
    LOGGER << "Input database..." << std::endl;
    input_db->printClassData(std::cout);
    LOGGER << "\nVariable database..." << std::endl;
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(std::cout);

    LOGGER << "\nCheck Linear Advection data... " << std::endl;
    linear_advection_model->printClassData(std::cout);

    if (viz_dump_data && time_integrator->getIntegratorStep() % viz_dump_interval == 0)
    {
        visit_data_writer->writePlotData(
                patch_hierarchy,
                time_integrator->getIntegratorStep(),
                time_integrator->getIntegratorTime());
    }

    /*
     * Time step loop.  Note that the step count and integration
     * time are maintained by SAMRAI::algs::TimeRefinementIntegrator.
     */

    double loop_time = time_integrator->getIntegratorTime();
    double loop_time_end = time_integrator->getEndTime();

    int iteration_num = time_integrator->getIntegratorStep();
}

void SAMRAIWrapperContext::run(Real dt, int level)
{
    Real loop_time = 0;
    Real loop_time_end = dt;
    Real dt_now = dt;
    while ((loop_time < loop_time_end) && time_integrator->stepsRemaining())
    {

        int iteration_num = time_integrator->getIntegratorStep() + 1;

        MESSAGE << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        MESSAGE << "At begining of timestep # " << iteration_num - 1 << std::endl;
        MESSAGE << "Simulation time is " << loop_time << std::endl;

        Real dt_new = (Real) time_integrator->advanceHierarchy(dt_now);

        loop_time += dt_now;
        dt_now = dt_new;

        MESSAGE << "At end of timestep # " << iteration_num - 1 << std::endl;
        MESSAGE << "Simulation time is " << loop_time << std::endl;
        MESSAGE << "++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

        /*
         * At specified intervals, write restart and visualization files.
         */
//        if (write_restart)
//        {
//
//            if ((iteration_num % restart_interval) == 0)
//            {
//                SAMRAI::tbox::RestartManager::getManager()->writeRestartFile(restart_write_dirname, iteration_num);
//            }
//        }

        /*
         * At specified intervals, write out data files for plotting.
         */

//        if (viz_dump_data)
//        {
//            if ((iteration_num % viz_dump_interval) == 0)
//            {
//                visit_data_writer->writePlotData(patch_hierarchy, iteration_num, loop_time);
//            }
//        }


    }
}

void SAMRAIWrapperContext::teardown()
{
    /*
     * At conclusion of simulation, deallocate objects.
     */

    visit_data_writer.reset();
    time_integrator.reset();
    gridding_algorithm.reset();
    load_balancer.reset();
    box_generator.reset();
    error_detector.reset();
    hyp_level_integrator.reset();
    linear_advection_model.reset();
    patch_hierarchy.reset();
    grid_geometry.reset();

    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();

}

std::shared_ptr<simulation::ContextBase> create_context(std::string const &name)
{
    return std::make_shared<SAMRAIWrapperContext>();
}

} //namespace simpla