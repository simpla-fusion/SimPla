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
#include <simpla/data/DataBase.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/DataBlock.h>
#include <simpla/mesh/Worker.h>
#include <simpla/simulation/TimeIntegrator.h>
#include <boost/shared_ptr.hpp>
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
#include <SAMRAI/pdat/NodeVariable.h>
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/CellVariable.h>

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

namespace simpla
{
struct SAMRAILevelIntegrator;

struct SAMRAITimeIntegrator;

std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w)
{
    return std::dynamic_pointer_cast<simulation::TimeIntegrator>(std::make_shared<SAMRAITimeIntegrator>(name, w));
}

class SAMRAILevelIntegrator :
        public SAMRAI::algs::TimeRefinementLevelStrategy,
        public SAMRAI::mesh::StandardTagAndInitStrategy,
        public concept::Printable
{


public:

    SAMRAILevelIntegrator(std::shared_ptr<mesh::Worker> const &w,
                          boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom);

    /**
     * The destructor for SAMRAILevelIntegrator does nothing.
     */
    ~SAMRAILevelIntegrator();

    /**
      *       concept::Printable
      **/
    virtual std::string name() const { return m_name_; };

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    /**
     *        public SAMRAI::algs::TimeRefinementLevelStrategy,
     **/

    virtual void
    initializeLevelIntegrator(const boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_alg);

    virtual double getLevelDt(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level, const double dt_time,
                              const bool initial_time);

    virtual double
    getMaxFinerLevelDt(const int finer_level_number, const double coarse_dt, const SAMRAI::hier::IntVector &ratio);

    virtual double advanceLevel(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
                                const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                const double current_time, const double new_time, const bool first_step,
                                const bool last_step, const bool regrid_advance = false);

    virtual void standardLevelSynchronization(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                              const int coarsest_level, const int finest_level, const double sync_time,
                                              const std::vector<double> &old_times);

    virtual void
    synchronizeNewLevels(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int coarsest_level,
                         const int finest_level, const double sync_time, const bool initial_time);

    virtual void resetTimeDependentData(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level, const double new_time,
                                        const bool can_be_refined);

    virtual void resetDataToPreadvanceState(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level);

    virtual bool usingRefinedTimestepping() const;


    /**
     *
     *         public SAMRAI::mesh::StandardTagAndInitStrategy,
     **/
//    virtual double 	getLevelDt (const boost::shared_ptr< SAMRAI:: hier::PatchLevel > &level, const double dt_time, const bool initial_time);
//
//    virtual double 	advanceLevel (const boost::shared_ptr< SAMRAI:: hier::PatchLevel > &level, const boost::shared_ptr< SAMRAI:: hier::PatchHierarchy > &hierarchy, const double current_time, const double new_time, const bool first_step, const bool last_step, const bool regrid_advance=false);
//
//    virtual void 	resetTimeDependentData (const boost::shared_ptr< SAMRAI:: hier::PatchLevel > &level, const double new_time, const bool can_be_refined);
//
//    virtual void 	resetDataToPreadvanceState (const boost::shared_ptr< SAMRAI:: hier::PatchLevel > &level);

    virtual void
    initializeLevelData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int level_number,
                        const double init_data_time, const bool can_be_refined, const bool initial_time,
                        const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level = boost::shared_ptr<SAMRAI::hier::PatchLevel>(),
                        const bool allocate_data = true);

    virtual void resetHierarchyConfiguration(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                             const int coarsest_level, const int finest_level);

    virtual void
    applyGradientDetector(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int level_number,
                          const double error_data_time, const int tag_index, const bool initial_time,
                          const bool uses_richardson_extrapolation_too);

    virtual void
    applyRichardsonExtrapolation(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level, const double error_data_time,
                                 const int tag_index, const double deltat, const int error_coarsen_ratio,
                                 const bool initial_time, const bool uses_gradient_detector_too);

    virtual void coarsenDataForRichardsonExtrapolation(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                                       const int level_number,
                                                       const boost::shared_ptr<SAMRAI::hier::PatchLevel> &coarser_level,
                                                       const double coarsen_data_time, const bool before_advance);

    virtual void processHierarchyBeforeAddingNewLevel(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                                      const int level_number,
                                                      const boost::shared_ptr<SAMRAI::hier::BoxLevel> &new_box_level);


    virtual void
    processLevelBeforeRemoval(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int level_number,
                              const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level = boost::shared_ptr<SAMRAI::hier::PatchLevel>());

    virtual void
    checkUserTagData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int level_number,
                     const int tag_index) const;


    virtual void
    checkNewLevelTagData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy, const int level_number,
                         const int tag_index) const;


    /**
    *
    *
    **/
    boost::shared_ptr<SAMRAI::hier::VariableContext> getPlotContext() const { return d_current; }

    void registerModelVariables();

    void setupLoadBalancer();

    void initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time, const bool initial_time);


    void registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer);

    int ndims() const { return m_ndims_; }

private:
    std::shared_ptr<mesh::Worker> m_worker_;

    int m_ndims_ = 3;
    /*
     * The object name is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string m_name_;


    /*
     * We cache pointers to the grid geometry object to set up initial
     * data, set physical boundary conditions, and register plot
     * variables.
     */
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> d_grid_geometry = nullptr;

    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer = nullptr;

    /*
     * Data items used for nonuniform load balance, if used.
     */
    boost::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_workload_variable;

    int d_workload_data_id;
    boost::shared_ptr<SAMRAI::hier::VariableContext> d_scratch;
    boost::shared_ptr<SAMRAI::hier::VariableContext> d_current;
    boost::shared_ptr<SAMRAI::hier::VariableContext> d_new;

};

SAMRAILevelIntegrator::SAMRAILevelIntegrator(
        std::shared_ptr<mesh::Worker> const &w,
        boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom) :
        m_worker_(w), m_name_(w != nullptr ? w->name() : "unnamed"), d_grid_geometry(grid_geom),
        d_scratch(SAMRAI::hier::VariableDatabase::getDatabase()->getContext("SCRATCH")),
        d_current(SAMRAI::hier::VariableDatabase::getDatabase()->getContext("CURRENT")),
        d_new(SAMRAI::hier::VariableDatabase::getDatabase()->getContext("NEW"))
{
}

/*
 *************************************************************************
 *
 * Empty destructor for SAMRAILevelIntegrator class.
 *
 *************************************************************************
 */

SAMRAILevelIntegrator::~SAMRAILevelIntegrator()
{
}


void
SAMRAILevelIntegrator::initializeLevelIntegrator(
        const boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_alg)
{
    FUNCTION_START;
}

double
SAMRAILevelIntegrator::getLevelDt(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level, const double dt_time,
                                  const bool initial_time)
{
    return dt_time;
}

double
SAMRAILevelIntegrator::getMaxFinerLevelDt(const int finer_level_number, const double coarse_dt,
                                          const SAMRAI::hier::IntVector &ratio)
{
    return coarse_dt;
}

double
SAMRAILevelIntegrator::advanceLevel(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
                                    const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                    const double current_time, const double new_time,
                                    const bool first_step,
                                    const bool last_step, const bool regrid_advance)
{
    FUNCTION_START;
    return new_time;
}

void
SAMRAILevelIntegrator::standardLevelSynchronization(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                                    const int coarsest_level, const int finest_level,
                                                    const double sync_time,
                                                    const std::vector<double> &old_times)
{
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::synchronizeNewLevels(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                            const int coarsest_level,
                                            const int finest_level, const double sync_time, const bool initial_time)
{
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::resetTimeDependentData(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
                                              const double new_time,
                                              const bool can_be_refined)
{
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::resetDataToPreadvanceState(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level)
{
    FUNCTION_START;
}

bool
SAMRAILevelIntegrator::usingRefinedTimestepping() const { return true; }


void
SAMRAILevelIntegrator::initializeLevelData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                           const int level_number,
                                           const double init_data_time, const bool can_be_refined,
                                           const bool initial_time,
                                           const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level,
                                           const bool allocate_data)
{
    TBOX_ASSERT(hierarchy);
    TBOX_ASSERT((level_number >= 0) && (level_number <= hierarchy->getFinestLevelNumber()));
    TBOX_ASSERT(!old_level || level_number == old_level->getLevelNumber());
    TBOX_ASSERT(hierarchy->getPatchLevel(level_number));

    boost::shared_ptr<SAMRAI::hier::PatchLevel> level(hierarchy->getPatchLevel(level_number));

    const SAMRAI::tbox::SAMRAI_MPI &mpi(level->getBoxLevel()->getMPI());
    mpi.Barrier();

 /*
  * Allocate storage needed to initialize level and fill data
  * from coarser levels in AMR hierarchy, potentially. Since
  * time gets set when we allocate data, re-stamp it to current
  * time if we don't need to allocate.
  */
    if (allocate_data)
    {
        level->allocatePatchData(d_new_patch_init_data, init_data_time);
        level->allocatePatchData(d_old_time_dep_data, init_data_time);
    } else
    {
        level->setTime(init_data_time, d_new_patch_init_data);
    }

    /*
   * Create schedules for filling new level and fill data.
   */
    level->getBoxLevel()->getMPI().Barrier();

    if ((level_number > 0) || old_level)
    {
        t_init_level_create_sched->start();

        boost::shared_ptr<SAMRAI::xfer::RefineSchedule> sched(          d_fill_new_level->createSchedule(level,
                                                 old_level,
                                                 level_number - 1,
                                                 hierarchy,
                                                 d_patch_strategy));
        mpi.Barrier();
        t_init_level_create_sched->stop();

        d_patch_strategy->setDataContext(d_scratch);

        t_init_level_fill_data->start();
        sched->fillData(init_data_time);
        mpi.Barrier();
        t_init_level_fill_data->stop();

        d_patch_strategy->clearDataContext();
    }

    if ((d_number_time_data_levels == 3) && can_be_refined)
    {

        SAMRAI::hier::VariableDatabase *variable_db = SAMRAI::hier::VariableDatabase::getDatabase();

        for (SAMRAI::hier::PatchLevel::iterator ip(level->begin()); ip != level->end(); ++ip)
        {
            const boost::shared_ptr<SAMRAI::hier::Patch> &patch = *ip;

            std::list<boost::shared_ptr<SAMRAI::hier::Variable> >::iterator time_dep_var = d_time_dep_variables.begin();
            while (time_dep_var != d_time_dep_variables.end())
            {
                int old_indx = variable_db->mapVariableAndContextToIndex(*time_dep_var, d_old);
                int cur_indx = variable_db->mapVariableAndContextToIndex(*time_dep_var, d_current);

                patch->setPatchData(old_indx, patch->getPatchData(cur_indx));

                ++time_dep_var;
            }

        }

    }

    mpi.Barrier();
    t_init_level_fill_interior->start();
    /*
   * Initialize data on patch interiors.
   */
    d_patch_strategy->setDataContext(d_current);
    for (SAMRAI::hier::PatchLevel::iterator p(level->begin()); p != level->end(); ++p)
    {
        const boost::shared_ptr<SAMRAI::hier::Patch> &patch = *p;

        patch->allocatePatchData(d_temp_var_scratch_data, init_data_time);

        d_patch_strategy->initializeDataOnPatch(*patch, init_data_time, initial_time);

        patch->deallocatePatchData(d_temp_var_scratch_data);
    }
    d_patch_strategy->clearDataContext();
    mpi.Barrier();
    t_init_level_fill_interior->stop();

    t_initialize_level_data->stop();


    registerModelVariables();
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::resetHierarchyConfiguration(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                                   const int coarsest_level, const int finest_level)
{
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::applyGradientDetector(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                             const int level_number,
                                             const double error_data_time, const int tag_index, const bool initial_time,
                                             const bool uses_richardson_extrapolation_too)
{
    FUNCTION_START;
}

void
SAMRAILevelIntegrator::applyRichardsonExtrapolation(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
                                                    const double error_data_time,
                                                    const int tag_index, const double deltat,
                                                    const int error_coarsen_ratio,
                                                    const bool initial_time, const bool uses_gradient_detector_too) {}

void
SAMRAILevelIntegrator::coarsenDataForRichardsonExtrapolation(
        const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
        const int level_number,
        const boost::shared_ptr<SAMRAI::hier::PatchLevel> &coarser_level,
        const double coarsen_data_time, const bool before_advance) {}

void
SAMRAILevelIntegrator::processHierarchyBeforeAddingNewLevel(
        const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
        const int level_number,
        const boost::shared_ptr<SAMRAI::hier::BoxLevel> &new_box_level) {}


void
SAMRAILevelIntegrator::processLevelBeforeRemoval(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                                 const int level_number,
                                                 const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level) {}

void
SAMRAILevelIntegrator::checkUserTagData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                        const int level_number,
                                        const int tag_index) const {}


void
SAMRAILevelIntegrator::checkNewLevelTagData(const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                            const int level_number,
                                            const int tag_index) const {}


namespace detail
{

std::string get_visit_variable_type(std::type_info const &t_info)
{

    if (t_info == typeid(float) || t_info == typeid(double) || t_info == typeid(int)) { return "SCALAR"; }
    else if (t_info == typeid(nTuple<float, 3>) ||
             t_info == typeid(nTuple<double, 3>) ||
             t_info == typeid(nTuple<int, 3>) ||
             t_info == typeid(nTuple<long, 3>)
            ) { return "VECTOR"; }
//    else if (t_info == typeid(nTuple<float, 3, 3>) ||
//             t_info == typeid(nTuple<double, 3, 3>) ||
//             t_info == typeid(nTuple<int, 3, 3>) ||
//             t_info == typeid(nTuple<long, 3, 3>)
//            ) { return "TENSOR"; }
    else
    {
        UNIMPLEMENTED;
    }


}


template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Node(mesh::Attribute *item,
                       SAMRAILevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{

    auto var = boost::make_shared<SAMRAI::pdat::NodeVariable<T>>(SAMRAI::tbox::Dimension(integrator->ndims()),
                                                                 item->name());

    d_visit_writer->registerPlotQuantity(item->name(), get_visit_variable_type(item->value_type_info()),
                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}

template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Edge(mesh::Attribute *item,
                       SAMRAILevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(3);

    auto var = boost::make_shared<SAMRAI::pdat::EdgeVariable<T>>(d_dim, item->name());

    d_visit_writer->registerPlotQuantity(item->name(), get_visit_variable_type(item->value_type_info()),
                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()), 0, 1.0,
                                         "CELL");


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}

template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Face(mesh::Attribute *item,
                       SAMRAILevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(3);

    auto var = boost::make_shared<SAMRAI::pdat::FaceVariable<T>>(d_dim, item->name());

    d_visit_writer->registerPlotQuantity(item->name(), get_visit_variable_type(item->value_type_info()),
                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()), 0, 1.0,
                                         "CELL");


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}


template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Cell(mesh::Attribute *item,
                       SAMRAILevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(3);

    auto var = boost::make_shared<SAMRAI::pdat::CellVariable<T>>(d_dim, item->name());

    d_visit_writer->registerPlotQuantity(item->name(), get_visit_variable_type(item->value_type_info()),
                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}

template<typename TV, typename ...Args> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_t(mesh::Attribute *item, Args &&...args)
{
    boost::shared_ptr<SAMRAI::hier::Variable> var;

    if (item->entity_type() == mesh::VERTEX)
    {
        var = register_variable_Node<TV>(item, std::forward<Args>(args)...);
    } else if (item->entity_type() == mesh::EDGE)
    {
        var = register_variable_Edge<TV>(item, std::forward<Args>(args)...);
    } else if (item->entity_type() == mesh::FACE)
    {
        var = register_variable_Face<TV>(item, std::forward<Args>(args)...);
    } else if (item->entity_type() == mesh::VOLUME)
    {
        var = register_variable_Cell<TV>(item, std::forward<Args>(args)...);
    } else { UNIMPLEMENTED; }
    return var;
}

template<typename ...Args> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable(mesh::Attribute *item, Args &&...args)
{

    boost::shared_ptr<SAMRAI::hier::Variable> var;

    if (item->value_type_info() == typeid(float))
    {
        var = register_variable_t<float>(item, std::forward<Args>(args)...);
    } else if (item->value_type_info() == typeid(double))
    {
        var = register_variable_t<double>(item, std::forward<Args>(args)...);
    } else if (item->value_type_info() == typeid(int))
    {
        var = register_variable_t<int>(item, std::forward<Args>(args)...);
    } else if (item->value_type_info() == typeid(long))
    {
        var = register_variable_t<int>(item, std::forward<Args>(args)...);
    } else { RUNTIME_ERROR << "Unsupported m_value_ type" << std::endl; }


    return var;
};

}//namespace detail{
/*
 *************************************************************************
 *
 * Register conserved variable (u) (i.e., solution state variable) and
 * flux variable with hyperbolic integrator that manages storage for
 * those quantities.  Also, register plot data with VisIt.
 *
 *************************************************************************
 */

void SAMRAILevelIntegrator::registerModelVariables()
{
    FUNCTION_START;

    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

//    integrator->registerVariable(d_uval, d_nghosts,
//                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
//                                 d_grid_geometry,
//                                 "CONSERVATIVE_COARSEN",
//                                 "CONSERVATIVE_LINEAR_REFINE");
//
//    integrator->registerVariable(d_flux, d_fluxghosts,
//                                 SAMRAI::algs::HyperbolicLevelIntegrator::FLUX,
//                                 d_grid_geometry,
//                                 "CONSERVATIVE_COARSEN",
//                                 "NO_REFINE");
//    d_visit_writer->registerPlotQuantity("U",
//                                         "SCALAR",
//                                         vardb->mapVariableAndContextToIndex(
//                                                 d_uval, integrator->getPlotContext()));

    if (!d_visit_writer)
    {
        RUNTIME_ERROR << name() << ": registerModelVariables() VisIt data writer was not registered."
                "Consequently, no plot data will be written." << std::endl;
    }

    m_worker_->apply(
            [&](mesh::Worker::Observer &ob)
            {
                mesh::Attribute *item = ob.attribute();
                if (item == nullptr) { return; }

                auto var = detail::register_variable(item,
                                                     this,
                                                     d_visit_writer,
                                                     vardb
                );
            }
    );
    vardb->printClassData(std::cout);

}


/*
 *************************************************************************
 *
 * Set up parameters for nonuniform load balancing, if used.
 *
 *************************************************************************
 */

void SAMRAILevelIntegrator::setupLoadBalancer()
{
//    bool d_use_nonuniform_workload = true;
//
//    const SAMRAI::hier::IntVector &zero_vec = SAMRAI::hier::IntVector::getZero(d_dim);
//
//    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();
//
//    if (d_use_nonuniform_workload && gridding_algorithm)
//    {
//        auto load_balancer = boost::dynamic_pointer_cast<SAMRAI::mesh::CascadePartitioner>(
//                gridding_algorithm->getLoadBalanceStrategy());
//
//        if (load_balancer)
//        {
//            d_workload_variable.reset(new SAMRAI::pdat::CellVariable<double>(d_dim, "workload_variable", 1));
//            d_workload_data_id = vardb->registerVariableAndContext(d_workload_variable,
//                                                                   vardb->getContext("WORKLOAD"),
//                                                                   zero_vec);
//            load_balancer->setWorkloadPatchDataIndex(d_workload_data_id);
//        } else
//        {
//            WARNING << d_object_name << ": "
//                    << "  Unknown load balancer used in gridding algorithm."
//                    << "  Ignoring request for nonuniform load balancing." << std::endl;
//            d_use_nonuniform_workload = false;
//        }
//    } else
//    {
//        d_use_nonuniform_workload = false;
//    }

}

/*
 *************************************************************************
 *
 * Set initial data for solution variables on patch interior.
 * This routine is called whenever a new patch is introduced to the
 * AMR patch hierarchy.  Note that the routine does nothing unless
 * we are at the initial time.  In all other cases, conservative
 * interpolation from coarser levels and copies from patches at the
 * same mesh resolution are sufficient to set data.
 *
 *************************************************************************
 */
void SAMRAILevelIntegrator::initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time,
                                                  const bool initial_time)
{
//    FUNCTION_START;
//    {
//        nTuple<size_type, 3> lo, up;
//        auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
//        lo = pgeom->getXLower();
//        up = pgeom->getXUpper();
//        INFORM << "initializeDataOnPatch" << " initial_time = " << std::boolalpha << initial_time << " level= "
//               << patch.getPatchLevelNumber() << " box= [" << lo << up << "]" << std::endl;
//    }
//    if (initial_time)
//    {
//
//        auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
//        TBOX_ASSERT(pgeom);
//        const double *dx = pgeom->getDx();
//        const double *xlo = pgeom->getXLower();
//        const double *xhi = pgeom->getXUpper();
//
//        auto uval = (boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
//                patch.getPatchData(d_uval, getDataContext())));
//
//        TBOX_ASSERT(uval);
//
//        SAMRAI::hier::IntVector ghost_cells(uval->getGhostCellWidth());
//
//        const SAMRAI::hier::Index ifirst = patch.getBox().lower();
//        const SAMRAI::hier::Index ilast = patch.getBox().upper();
//
//        if (d_data_problem_int == SPHERE)
//        {
//            auto p = uval->getPointer();
//
//            if (d_dim == SAMRAI::tbox::Dimension(2))
//            {
////                SAMRAI_F77_FUNC(initsphere2d, INITSPHERE2D)(d_data_problem_int, dx, xlo,
////                                                            xhi,
////                                                            ifirst(0), ilast(0),
////                                                            ifirst(1), ilast(1),
////                                                            ghost_cells(0),
////                                                            ghost_cells(1),
////                                                            uval->getPointer(),
////                                                            d_uval_inside,
////                                                            d_uval_outside,
////                                                            &d_center[0], d_radius);
//            }
//            if (d_dim == SAMRAI::tbox::Dimension(3))
//            {
////                SAMRAI_F77_FUNC(initsphere3d, INITSPHERE3D)(d_data_problem_int, dx, xlo,
////                                                            xhi,
////                                                            ifirst(0), ilast(0),
////                                                            ifirst(1), ilast(1),
////                                                            ifirst(2), ilast(2),
////                                                            ghost_cells(0),
////                                                            ghost_cells(1),
////                                                            ghost_cells(2),
////                                                            uval->getPointer(),
////                                                            d_uval_inside,
////                                                            d_uval_outside,
////                                                            &d_center[0], d_radius);
//            }
//
//        } else if (d_data_problem_int == SINE_CONSTANT_X ||
//                   d_data_problem_int == SINE_CONSTANT_Y ||
//                   d_data_problem_int == SINE_CONSTANT_Z)
//        {
//
//            const double *domain_xlo = d_grid_geometry->getXLower();
//            const double *domain_xhi = d_grid_geometry->getXUpper();
//            std::vector<double> domain_length(d_dim.getValue());
//            for (int i = 0; i < d_dim.getValue(); ++i)
//            {
//                domain_length[i] = domain_xhi[i] - domain_xlo[i];
//            }
//
//            if (d_dim == SAMRAI::tbox::Dimension(2))
//            {
////                SAMRAI_F77_FUNC(linadvinitsine2d, LINADVINITSINE2D)(d_data_problem_int,
////                                                                    dx, xlo,
////                                                                    domain_xlo, &domain_length[0],
////                                                                    ifirst(0), ilast(0),
////                                                                    ifirst(1), ilast(1),
////                                                                    ghost_cells(0),
////                                                                    ghost_cells(1),
////                                                                    uval->getPointer(),
////                                                                    d_number_of_intervals,
////                                                                    &d_front_position[0],
////                                                                    &d_interval_uval[0],
////                                                                    d_amplitude,
////                                                                    &d_frequency[0]);
//            }
//            if (d_dim == SAMRAI::tbox::Dimension(3))
//            {
////                SAMRAI_F77_FUNC(linadvinitsine3d, LINADVINITSINE3D)(d_data_problem_int,
////                                                                    dx, xlo,
////                                                                    domain_xlo, &domain_length[0],
////                                                                    ifirst(0), ilast(0),
////                                                                    ifirst(1), ilast(1),
////                                                                    ifirst(2), ilast(2),
////                                                                    ghost_cells(0),
////                                                                    ghost_cells(1),
////                                                                    ghost_cells(2),
////                                                                    uval->getPointer(),
////                                                                    d_number_of_intervals,
////                                                                    &d_front_position[0],
////                                                                    &d_interval_uval[0],
////                                                                    d_amplitude,
////                                                                    &d_frequency[0]);
//            }
//        } else
//        {
//
//            if (d_dim == SAMRAI::tbox::Dimension(2))
//            {
////                SAMRAI_F77_FUNC(linadvinit2d, LINADVINIT2D)(d_data_problem_int, dx, xlo,xhi,
////                                                            ifirst(0), ilast(0),
////                                                            ifirst(1), ilast(1),
////                                                            ghost_cells(0),
////                                                            ghost_cells(1),
////                                                            uval->getPointer(),
////                                                            d_number_of_intervals,
////                                                            &d_front_position[0],
////                                                            &d_interval_uval[0]);
//            }
//            if (d_dim == SAMRAI::tbox::Dimension(3))
//            {
////                SAMRAI_F77_FUNC(linadvinit3d, LINADVINIT3D)(d_data_problem_int, dx, xlo,
////                                                            xhi,
////                                                            ifirst(0), ilast(0),
////                                                            ifirst(1), ilast(1),
////                                                            ifirst(2), ilast(2),
////                                                            ghost_cells(0),
////                                                            ghost_cells(1),
////                                                            ghost_cells(2),
////                                                            uval->getPointer(),
////                                                            d_number_of_intervals,
////                                                            &d_front_position[0],
////                                                            &d_interval_uval[0]);
//            }
//        }
//
//    }
//
//    if (d_use_nonuniform_workload)
//    {
//        if (!patch.checkAllocated(d_workload_data_id))
//        {
//            patch.allocatePatchData(d_workload_data_id);
//        }
//        boost::shared_ptr<SAMRAI::pdat::CellData<double>> workload_data(
//                boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
//                        patch.getPatchData(d_workload_data_id)));
//        TBOX_ASSERT(workload_data);
//
//        const SAMRAI::hier::Box &box = patch.getBox();
//        const SAMRAI::hier::BoxId &box_id = box.getBoxId();
//        const SAMRAI::hier::LocalId &local_id = box_id.getLocalId();
//        double id_val = local_id.getValue() % 2 ? static_cast<double>(local_id.getValue() % 10) : 0.0;
//        workload_data->fillAll(1.0 + id_val);
//    }

}


/*
 *************************************************************************
 *
 * Register VisIt data writer to write data to plot files that may
 * be postprocessed by the VisIt tool.
 *
 *************************************************************************
 */


void SAMRAILevelIntegrator::registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer)
{
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}


/*
 *************************************************************************
 *
 * Write SAMRAILevelIntegrator object state to specified stream.
 *
 *************************************************************************
 */

std::ostream &SAMRAILevelIntegrator::print(std::ostream &os, int indent) const
{
//    int j, k;
//
//    os << "\nSAMRAILevelIntegrator::printClassData..." << std::endl;
//    os << "SAMRAILevelIntegrator: this = " << (SAMRAILevelIntegrator *)
//            this << std::endl;
//    os << "d_object_name = " << d_object_name << std::endl;
//    os << "d_grid_geometry = "
//       << d_grid_geometry.get() << std::endl;
//
//    os << "Parameters for numerical method ..." << std::endl;
//    os << "   d_advection_velocity = ";
//    for (j = 0; j < d_dim.getValue(); ++j) os << d_advection_velocity[j] << " ";
//    os << std::endl;
//    os << "   d_godunov_order = " << d_godunov_order << std::endl;
//    os << "   d_corner_transport = " << d_corner_transport << std::endl;
//    os << "   d_nghosts = " << d_nghosts << std::endl;
//    os << "   d_fluxghosts = " << d_fluxghosts << std::endl;
//
//    os << "Problem description and initial data..." << std::endl;
//    os << "   d_data_problem = " << d_data_problem << std::endl;
//    os << "   d_data_problem_int = " << d_data_problem << std::endl;
//
//    os << "       d_radius = " << d_radius << std::endl;
//    os << "       d_center = ";
//    for (j = 0; j < d_dim.getValue(); ++j) os << d_center[j] << " ";
//    os << std::endl;
//    os << "       d_uval_inside = " << d_uval_inside << std::endl;
//    os << "       d_uval_outside = " << d_uval_outside << std::endl;
//
//    os << "       d_number_of_intervals = " << d_number_of_intervals << std::endl;
//    os << "       d_front_position = ";
//    for (k = 0; k < d_number_of_intervals - 1; ++k)
//    {
//        os << d_front_position[k] << "  ";
//    }
//    os << std::endl;
//    os << "       d_interval_uval = " << std::endl;
//    for (k = 0; k < d_number_of_intervals; ++k)
//    {
//        os << "            " << d_interval_uval[k] << std::endl;
//    }
//    os << "   Boundary condition data " << std::endl;
//
//    if (d_dim == SAMRAI::tbox::Dimension(2))
//    {
//        for (j = 0; j < static_cast<int>(d_scalar_bdry_edge_conds.size()); ++j)
//        {
//            os << "       d_scalar_bdry_edge_conds[" << j << "] = "
//               << d_scalar_bdry_edge_conds[j] << std::endl;
//            if (d_scalar_bdry_edge_conds[j] == BdryCond::DIRICHLET)
//            {
//                os << "         d_bdry_edge_uval[" << j << "] = "
//                   << d_bdry_edge_uval[j] << std::endl;
//            }
//        }
//        os << std::endl;
//        for (j = 0; j < static_cast<int>(d_scalar_bdry_node_conds.size()); ++j)
//        {
//            os << "       d_scalar_bdry_node_conds[" << j << "] = "
//               << d_scalar_bdry_node_conds[j] << std::endl;
//            os << "       d_node_bdry_edge[" << j << "] = "
//               << d_node_bdry_edge[j] << std::endl;
//        }
//    }
//    if (d_dim == SAMRAI::tbox::Dimension(3))
//    {
//        for (j = 0; j < static_cast<int>(d_scalar_bdry_face_conds.size()); ++j)
//        {
//            os << "       d_scalar_bdry_face_conds[" << j << "] = "
//               << d_scalar_bdry_face_conds[j] << std::endl;
//            if (d_scalar_bdry_face_conds[j] == BdryCond::DIRICHLET)
//            {
//                os << "         d_bdry_face_uval[" << j << "] = "
//                   << d_bdry_face_uval[j] << std::endl;
//            }
//        }
//        os << std::endl;
//        for (j = 0; j < static_cast<int>(d_scalar_bdry_edge_conds.size()); ++j)
//        {
//            os << "       d_scalar_bdry_edge_conds[" << j << "] = "
//               << d_scalar_bdry_edge_conds[j] << std::endl;
//            os << "       d_edge_bdry_face[" << j << "] = "
//               << d_edge_bdry_face[j] << std::endl;
//        }
//        os << std::endl;
//        for (j = 0; j < static_cast<int>(d_scalar_bdry_node_conds.size()); ++j)
//        {
//            os << "       d_scalar_bdry_node_conds[" << j << "] = "
//               << d_scalar_bdry_node_conds[j] << std::endl;
//            os << "       d_node_bdry_face[" << j << "] = "
//               << d_node_bdry_face[j] << std::endl;
//        }
//    }
//
//    os << "   Refinement criteria parameters " << std::endl;
//
//    for (j = 0; j < static_cast<int>(d_refinement_criteria.size()); ++j)
//    {
//        os << "       d_refinement_criteria[" << j << "] = "
//           << d_refinement_criteria[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_dev_tol.size()); ++j)
//    {
//        os << "       d_dev_tol[" << j << "] = "
//           << d_dev_tol[j] << std::endl;
//    }
//    for (j = 0; j < static_cast<int>(d_dev.size()); ++j)
//    {
//        os << "       d_dev[" << j << "] = "
//           << d_dev[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_dev_time_max.size()); ++j)
//    {
//        os << "       d_dev_time_max[" << j << "] = "
//           << d_dev_time_max[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_dev_time_min.size()); ++j)
//    {
//        os << "       d_dev_time_min[" << j << "] = "
//           << d_dev_time_min[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_grad_tol.size()); ++j)
//    {
//        os << "       d_grad_tol[" << j << "] = "
//           << d_grad_tol[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_grad_time_max.size()); ++j)
//    {
//        os << "       d_grad_time_max[" << j << "] = "
//           << d_grad_time_max[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_grad_time_min.size()); ++j)
//    {
//        os << "       d_grad_time_min[" << j << "] = "
//           << d_grad_time_min[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_shock_onset.size()); ++j)
//    {
//        os << "       d_shock_onset[" << j << "] = "
//           << d_shock_onset[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_shock_tol.size()); ++j)
//    {
//        os << "       d_shock_tol[" << j << "] = "
//           << d_shock_tol[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_shock_time_max.size()); ++j)
//    {
//        os << "       d_shock_time_max[" << j << "] = "
//           << d_shock_time_max[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_shock_time_min.size()); ++j)
//    {
//        os << "       d_shock_time_min[" << j << "] = "
//           << d_shock_time_min[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_rich_tol.size()); ++j)
//    {
//        os << "       d_rich_tol[" << j << "] = "
//           << d_rich_tol[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_rich_time_max.size()); ++j)
//    {
//        os << "       d_rich_time_max[" << j << "] = "
//           << d_rich_time_max[j] << std::endl;
//    }
//    os << std::endl;
//    for (j = 0; j < static_cast<int>(d_rich_time_min.size()); ++j)
//    {
//        os << "       d_rich_time_min[" << j << "] = "
//           << d_rich_time_min[j] << std::endl;
//    }
//    os << std::endl;

}


struct SAMRAITimeIntegrator : public simulation::TimeIntegrator
{
    typedef simulation::TimeIntegrator base_type;
public:
    SAMRAITimeIntegrator(std::string const &s, std::shared_ptr<mesh::Worker> const &w);

    ~SAMRAITimeIntegrator();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void load(data::DataBase const &);

    virtual void save(data::DataBase *) const;

    virtual void deploy();

    virtual void tear_down();

    virtual bool is_valid() const;

    virtual size_type step() const;

    virtual Real time_now() const;

    virtual void next_time_step(Real dt);

    virtual void check_point();

    virtual void register_worker(std::shared_ptr<mesh::Worker> const &w) { m_worker_ = w; }


private:
    bool m_is_valid_ = false;


    std::shared_ptr<mesh::Worker> m_worker_;

    boost::shared_ptr<SAMRAILevelIntegrator> level_integrator;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;


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

    static constexpr int ndims = 3;
};

SAMRAITimeIntegrator::SAMRAITimeIntegrator(std::string const &s, std::shared_ptr<mesh::Worker> const &w)
        : base_type(s), m_worker_(w)
{
    /*
      * Initialize SAMRAI::tbox::MPI.
      */
    SAMRAI::tbox::SAMRAI_MPI::init(0, nullptr);

    SAMRAI::tbox::SAMRAIManager::initialize();
    /*
     * Initialize SAMRAI, enable logging, and process command line.
     */
    SAMRAI::tbox::SAMRAIManager::startup();
//    const SAMRAI::tbox::SAMRAI_MPI & mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());
}

SAMRAITimeIntegrator::~SAMRAITimeIntegrator()
{
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();
}

std::ostream &SAMRAITimeIntegrator::print(std::ostream &os, int indent) const
{
    level_integrator->print(os, indent + 1);
    return os;
};


void SAMRAITimeIntegrator::load(data::DataBase const &db) { m_is_valid_ = false; }

void SAMRAITimeIntegrator::save(data::DataBase *) const { UNIMPLEMENTED; }

namespace detail
{
void convert_database_r(data::DataBase const &src, boost::shared_ptr<SAMRAI::tbox::Database> &dest,
                        std::string const &key = "")
{

    if (src.is_table())
    {
        auto sub_db = key == "" ? dest : dest->putDatabase(key);

        src.foreach([&](std::string const &k, data::DataBase const &v) { convert_database_r(v, sub_db, k); });
    } else if (key == "") { return; }
    else if (src.empty()) { dest->putDatabase(key); }
    else if (src.is_boolean()) { dest->putBool(key, src.as<bool>()); }
    else if (src.is_string()) { dest->putString(key, src.as<std::string>()); }
    else if (src.is_floating_point()) { dest->putDouble(key, src.as<double>()); }
    else if (src.is_integral()) { dest->putInteger(key, src.as<int>()); }
    else if (src.type() == typeid(nTuple<bool, 3>))
    {
        dest->putBoolArray(key, &src.as<nTuple<bool, 3 >>()[0], 3);
    } else if (src.type() == typeid(nTuple<int, 3>))
    {
        dest->putIntegerArray(key, &src.as<nTuple<int, 3 >>()[0], 3);
    } else if (src.type() == typeid(nTuple<double, 3>))
    {
        dest->putDoubleArray(key, &src.as<nTuple<double, 3 >>()[0], 3);
    }
//    else if (src.type() == typeid(box_type)) { dest->putDoubleArray(key, &src.as<box_type>()[0], 3); }
    else if (src.type() == typeid(index_box_type))
    {
        nTuple<int, 3> i_lo, i_up;
        std::tie(i_lo, i_up) = src.as<index_box_type>();
        SAMRAI::tbox::Dimension dim(3);
        dest->putDatabaseBox(key, SAMRAI::tbox::DatabaseBox(dim, &(i_lo[0]), &(i_up[0])));
    } else
    {
        WARNING << " Unknown type [" << src << "]" << std::endl;
    }

}

boost::shared_ptr<SAMRAI::tbox::Database>
convert_database(data::DataBase const &src, std::string const &s_name = "")
{
    auto dest = boost::dynamic_pointer_cast<SAMRAI::tbox::Database>(
            boost::make_shared<SAMRAI::tbox::MemoryDatabase>(s_name));
    convert_database_r(src, dest);
    return dest;
}
}//namespace detail{
void SAMRAITimeIntegrator::deploy()
{

    bool use_refined_timestepping = db["use_refined_timestepping"].template as<bool>(true);

    m_is_valid_ = true;

    SAMRAI::tbox::Dimension dim(ndims);

    auto samrai_cfg = detail::convert_database(db, name());



    /**
    * Create major algorithm and data objects which comprise application.
    * Each object will be initialized either from input data or restart
    * files, or a combination of both.  Refer to each class constructor
    * for details.  For more information on the composition of objects
    * for this application, see comments at top of file.
    */


    grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(dim, "CartesianGeometry",
                                                                            samrai_cfg->getDatabase(
                                                                                    "CartesianGeometry"));
//    grid_geometry->printClassData(std::cout);
    //---------------------------------

    patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>("PatchHierarchy", grid_geometry,
                                                                       samrai_cfg->getDatabase("PatchHierarchy"));
//    patch_hierarchy->recursivePrint(std::cout, "", 1);
    //---------------------------------

    level_integrator = boost::make_shared<SAMRAILevelIntegrator>(m_worker_, grid_geometry);

//    level_integrator->printClassData(std::cout);

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize", level_integrator.get(),
            samrai_cfg->getDatabase("StandardTagAndInitialize"));
    //---------------------------------

    /**
     *  create grid_algorithm
     */
    auto box_generator = boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(dim,
                                                                           samrai_cfg->getDatabase("BergerRigoutsos"));

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(dim, "LoadBalancer",
                                                                              samrai_cfg->getDatabase("LoadBalancer"));

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

//    load_balancer->printStatistics(std::cout);

    auto gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
            patch_hierarchy,
            "GriddingAlgorithm",
            samrai_cfg->getDatabase("GriddingAlgorithm"),
            error_detector,
            box_generator,
            load_balancer);

//    gridding_algorithm->printClassData(std::cout);
    //---------------------------------

    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            samrai_cfg->getDatabase("TimeRefinementIntegrator"),
            patch_hierarchy,
            level_integrator,
            gridding_algorithm);


    visit_data_writer = boost::make_shared<SAMRAI::appu::VisItDataWriter>(
            dim,
            db["output_writer_name"].as<std::string>(name() + " VisIt Writer"),
            db["output_dir_name"].as<std::string>(name()),
            db["visit_number_procs_per_file"].as<int>(1)
    );

    level_integrator->registerVisItDataWriter(visit_data_writer);


    time_integrator->initializeHierarchy();

//    time_integrator->printClassData(std::cout);

    MESSAGE << name() << " is deployed!" << std::endl;


    samrai_cfg->printClassData(std::cout);
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(std::cout);

};

void SAMRAITimeIntegrator::tear_down()
{
    m_is_valid_ = false;

    visit_data_writer.reset();

    time_integrator.reset();

    level_integrator.reset();

}


bool SAMRAITimeIntegrator::is_valid() const { return m_is_valid_; }

void SAMRAITimeIntegrator::next_time_step(Real dt)
{
    assert(is_valid());
    MESSAGE << " Time = " << time_now() << " Step = " << step() << std::endl;
    time_integrator->advanceHierarchy(dt, true);

}


void SAMRAITimeIntegrator::check_point()
{
    if (visit_data_writer != nullptr)
    {
        VERBOSE << visit_data_writer->getObjectName() << std::endl;

        visit_data_writer->writePlotData(patch_hierarchy,
                                         time_integrator->getIntegratorStep(),
                                         time_integrator->getIntegratorTime());
    }
}

Real SAMRAITimeIntegrator::time_now() const { return static_cast<Real>( time_integrator->getIntegratorTime()); }

size_type SAMRAITimeIntegrator::step() const { return static_cast<size_type>( time_integrator->getIntegratorStep()); }

/*
 *************************************************************************
 *
 * Routine to check boundary data when debugging.
 *
 *************************************************************************
 */
//
//void SAMRAILevelIntegrator::checkBoundaryData(
//        int btype,
//        const SAMRAI::hier::DataBlockBase &patch,
//        const SAMRAI::hier::IntVector &ghost_width_to_check,
//        const std::vector<int> &scalar_bconds) const
//{
//#ifdef DEBUG_CHECK_ASSERTIONS
//    if (d_dim == SAMRAI::tbox::Dimension(2))
//    {
//        TBOX_ASSERT(btype == Bdry::EDGE2D ||
//                    btype == Bdry::NODE2D);
//    }
//    if (d_dim == SAMRAI::tbox::Dimension(3))
//    {
//        TBOX_ASSERT(btype == Bdry::FACE3D ||
//                    btype == Bdry::EDGE3D ||
//                    btype == Bdry::NODE3D);
//    }
//#endif
//
//    const boost::shared_ptr<SAMRAI::geom::CartesianPatchGeometry> pgeom(
//             boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry, SAMRAI::hier::PatchGeometry>(
//                    patch.getPatchGeometry()));
//    TBOX_ASSERT(pgeom);
//    const std::vector<SAMRAI::hier::BoundaryBox> &bdry_boxes =
//            pgeom->getCodimensionBoundaries(btype);
//
//    SAMRAI::hier::VariableDatabase *vdb = SAMRAI::hier::VariableDatabase::getDatabase();
//
//    for (int i = 0; i < static_cast<int>(bdry_boxes.size()); ++i)
//    {
//        SAMRAI::hier::BoundaryBox bbox = bdry_boxes[i];
//        TBOX_ASSERT(bbox.getBoundaryType() == btype);
//        int bloc = bbox.getLocationIndex();
//
//        int bscalarcase = 0, refbdryloc = 0;
//        if (d_dim == SAMRAI::tbox::Dimension(2))
//        {
//            if (btype == Bdry::EDGE2D)
//            {
//                TBOX_ASSERT(static_cast<int>(scalar_bconds.size()) ==
//                            NUM_2D_EDGES);
//                bscalarcase = scalar_bconds[bloc];
//                refbdryloc = bloc;
//            } else
//            { // btype == Bdry::NODE2D
//                TBOX_ASSERT(static_cast<int>(scalar_bconds.size()) ==
//                            NUM_2D_NODES);
//                bscalarcase = scalar_bconds[bloc];
//                refbdryloc = d_node_bdry_edge[bloc];
//            }
//        }
//        if (d_dim == SAMRAI::tbox::Dimension(3))
//        {
//            if (btype == Bdry::FACE3D)
//            {
//                TBOX_ASSERT(static_cast<int>(scalar_bconds.size()) ==
//                            NUM_3D_FACES);
//                bscalarcase = scalar_bconds[bloc];
//                refbdryloc = bloc;
//            } else if (btype == Bdry::EDGE3D)
//            {
//                TBOX_ASSERT(static_cast<int>(scalar_bconds.size()) ==
//                            NUM_3D_EDGES);
//                bscalarcase = scalar_bconds[bloc];
//                refbdryloc = d_edge_bdry_face[bloc];
//            } else
//            { // btype == Bdry::NODE3D
//                TBOX_ASSERT(static_cast<int>(scalar_bconds.size()) ==
//                            NUM_3D_NODES);
//                bscalarcase = scalar_bconds[bloc];
//                refbdryloc = d_node_bdry_face[bloc];
//            }
//        }
//
//        int num_bad_values = 0;
//
//        if (d_dim == SAMRAI::tbox::Dimension(2))
//        {
//            num_bad_values =
//                    SAMRAI::appu::CartesianBoundaryUtilities2::checkBdryData(
//                            d_uval->getName(),
//                            patch,
//                            vdb->mapVariableAndContextToIndex(d_uval, getDataContext()), 0,
//                            ghost_width_to_check,
//                            bbox,
//                            bscalarcase,
//                            d_bdry_edge_uval[refbdryloc]);
//        }
//        if (d_dim == SAMRAI::tbox::Dimension(3))
//        {
//            num_bad_values =
//                    SAMRAI::appu::CartesianBoundaryUtilities3::checkBdryData(
//                            d_uval->getName(),
//                            patch,
//                            vdb->mapVariableAndContextToIndex(d_uval, getDataContext()), 0,
//                            ghost_width_to_check,
//                            bbox,
//                            bscalarcase,
//                            d_bdry_face_uval[refbdryloc]);
//        }
//
//
//    }
//
//}
//
//void
//SAMRAILevelIntegrator::checkUserTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const
//{
//    boost::shared_ptr<SAMRAI::pdat::CellData<int> > tags( boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int>, SAMRAI::hier::PatchData>(patch.getPatchData(tag_index)));
//    TBOX_ASSERT(tags);
//}
//
//void
//SAMRAILevelIntegrator::checkNewPatchTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const
//{
//    boost::shared_ptr<SAMRAI::pdat::CellData<int> > tags(
//             boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int>, SAMRAI::hier::PatchData>(
//                    patch.getPatchData(tag_index)));
//    TBOX_ASSERT(tags);
//}

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
//          public mesh::DataBlockBase<V, M, IFORM>
//{
//    typedef typename SAMRAITraitsPatch<V, IFORM>::type samari_base_type;
//    typedef mesh::DataBlockBase<V, M, IFORM> simpla_base_type;
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
//            samrai_base_type(SAMRAI::tbox::Dimension(M::NDIMS), name, 1), simpla_base_type(m) {}
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
//std::shared_ptr<mesh::Attribute>
//create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                      std::shared_ptr<mesh::Atlas> const &m, std::string const &name)
//{
//}
//
//
//std::shared_ptr<mesh::Attribute>
//create_attribute_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                      std::shared_ptr<mesh::MeshBlock> const &m, std::string const &name)
//{
//}
//
//std::shared_ptr<mesh::DataEntityHeavy>
//create_patch_impl(std::type_info const &type_info, std::type_info const &mesh_info, mesh::MeshEntityType const &,
//                  std::shared_ptr<mesh::MeshBlock> const &m)
//{
//
//}
//}//namespace detail
//} //namespace simpla