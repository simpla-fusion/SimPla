//
// Created by salmon on 16-10-27.
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


#include "SAMRAIContext.h"

namespace simpla { namespace simulation
{
struct SAMRAIWorker : public SAMRAI::algs::HyperbolicPatchStrategy,
                      public SAMRAI::appu::BoundaryUtilityStrategy
{

    std::shared_ptr<WorkerBase> m_worker_;
//
//
//    /**
//     *  implementations of function in  SAMRAI::algs::TimeRefinementLevelStrategy
//     * @{
//     */
//    virtual void
//    initializeLevelIntegrator(const boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_alg) = 0;
//
//
//    virtual double
//    getLevelDt(
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
//            const double dt_time,
//            const bool initial_time);
//
//
//    virtual double
//    getMaxFinerLevelDt(
//            const int finer_level_number,
//            const double coarse_dt,
//            const SAMRAI::hier::IntVector &ratio);
//
//    virtual double
//    advanceLevel(
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const double current_time,
//            const double new_time,
//            const bool first_step,
//            const bool last_step,
//            const bool regrid_advance = false);
//
//
//    virtual void
//    standardLevelSynchronization(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int coarsest_level,
//            const int finest_level,
//            const double sync_time,
//            const std::vector<double> &old_times);
//
//
//    virtual void
//    synchronizeNewLevels(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int coarsest_level,
//            const int finest_level,
//            const double sync_time,
//            const bool initial_time);
//
//
//    virtual void
//    resetTimeDependentData(
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
//            const double new_time,
//            const bool can_be_refined);
//
//
//    virtual void
//    resetDataToPreadvanceState(
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level);
//
//
//    virtual bool
//    usingRefinedTimestepping() const;
//
//    /**
//     * @}
//     */
//
//    /**
//     *  implementations of function in  StandardTagAndInitStrategy
//     * @{
//     */
//    void initializeLevelData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//                             const int level_number,
//                             const double init_data_time,
//                             const bool can_be_refined,
//                             const bool initial_time,
//                             const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level = boost::shared_ptr<SAMRAI::hier::PatchLevel>(),
//                             const bool allocate_data = true);
//
//    virtual void
//    resetHierarchyConfiguration(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int coarsest_level,
//            const int finest_level) = 0;
//
//    virtual void
//    applyGradientDetector(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int level_number,
//            const double error_data_time,
//            const int tag_index,
//            const bool initial_time,
//            const bool uses_richardson_extrapolation_too);
//
//    virtual void
//    applyRichardsonExtrapolation(
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
//            const double error_data_time,
//            const int tag_index,
//            const double deltat,
//            const int error_coarsen_ratio,
//            const bool initial_time,
//            const bool uses_gradient_detector_too);
//
//    virtual void
//    coarsenDataForRichardsonExtrapolation(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int level_number,
//            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &coarse_level,
//            const double coarsen_data_time,
//            const bool before_advance);
//
//
//    /**
//    * @}
//    */
};

struct SAMRAIContext::pimpl_s
{

    boost::shared_ptr<SAMRAIWorker> m_levelIntegrator;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;

    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;
//
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

SAMRAIContext::SAMRAIContext() {};

SAMRAIContext::~SAMRAIContext() {};

void SAMRAIContext::setup(int argc, char *argv[])
{
    boost::shared_ptr<SAMRAI::tbox::InputDatabase> input_db(new SAMRAI::tbox::InputDatabase("input_db"));

    const SAMRAI::tbox::Dimension dim(static_cast<unsigned short>(input_db->getInteger("dim")));


    /*
     * Create major algorithm and data objects which comprise application.
     * Each object will be initialized either from input data or restart
     * files, or a combination of both.  Refer to each class constructor
     * for details.  For more information on the composition of objects
     * for this application, see comments at top of file.
     */

    m_pimpl_->grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(
            dim,
            "CartesianGeometry",
            input_db->getDatabase("CartesianGeometry"));

    auto patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>(
            "PatchHierarchy",
            m_pimpl_->grid_geometry,
            input_db->getDatabase("PatchHierarchy"));


    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize",
            m_pimpl_->m_levelIntegrator.get(),
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
            m_pimpl_->patch_hierarchy,
            "GriddingAlgorithm",
            input_db->getDatabase("GriddingAlgorithm"),
            error_detector,
            box_generator,
            load_balancer);
    //---------------------------------
    m_pimpl_->time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            input_db->getDatabase("TimeRefinementIntegrator"),
            patch_hierarchy,
            m_pimpl_->m_levelIntegrator,
            gridding_algorithm);

    //---------------------------------

    //---------------------------------


//    const std::string viz_dump_dirname = main_db->getStringWithDefault("viz_dump_dirname", base_name + ".visit");
//    int visit_number_procs_per_file = 1;
//
//    m_pimpl_->visit_data_writer = boost::make_shared<appu::VisItDataWriter>(
//            dim,
//            "LinAdv VisIt Writer",
//            viz_dump_dirname,
//            visit_number_procs_per_file);
//    patch_worker->registerVisItDataWriter(visit_data_writer);


};


void SAMRAIContext::registerWorker(std::string const &name, std::shared_ptr<WorkerBase> const &worker)
{
    m_pimpl_->m_levelIntegrator = boost::make_shared<SAMRAIWorker>(worker);

//    hyp_level_integrator = boost::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
//            "HyperbolicLevelIntegrator",
//            input_db->getDatabase("HyperbolicLevelIntegrator"),
//            patch_worker.get(),
//            use_refined_timestepping);
};

void SAMRAIContext::teardown() { m_pimpl_.release(); };


}}//namespace simpla { namespace simulation
