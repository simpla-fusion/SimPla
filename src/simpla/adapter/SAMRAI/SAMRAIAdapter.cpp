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

namespace simpla
{

class SAMRAIWorkerHyperbolic :
        public simulation::WorkerBase,
        public SAMRAI::algs::HyperbolicPatchStrategy
{
public:
    /**
     * The constructor for LinAdv sets default parameters for the linear
     * advection model.  Specifically, it creates variables that represent
     * the state of the solution.  The constructor also registers this
     * object for restart with the restart manager using the object name.
     *
     * After default values are set, this routine calls getFromRestart()
     * if execution from a restart file is specified.  Finally,
     * getFromInput() is called to read values from the given input
     * database (potentially overriding those found in the restart file).
     */
    SAMRAIWorkerHyperbolic(const std::string &object_name,
                           const SAMRAI::tbox::Dimension &dim,
                           boost::shared_ptr<SAMRAI::tbox::Database> input_db,
                           boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geom);

    /**
     * The destructor for LinAdv does nothing.
     */
    ~SAMRAIWorkerHyperbolic();

    ///
    ///  The following routines:
    ///
    ///      registerModelVariables(),
    ///      initializeDataOnPatch(),
    ///      computeStableDtOnPatch(),
    ///      computeFluxesOnPatch(),
    ///      conservativeDifferenceOnPatch(),
    ///      tagGradientDetectorCells(),
    ///      tagRichardsonExtrapolationCells()
    ///
    ///  are concrete implementations of functions declared in the
    ///  algs::HyperbolicPatchStrategy abstract base class.
    ///

    /**
     * Register LinAdv model variables with algs::HyperbolicLevelIntegrator
     * according to variable registration function provided by the integrator.
     * In other words, variables are registered according to their role
     * in the integration process (e.g., time-dependent, flux, etc.).
     * This routine also registers variables for plotting with the
     * Vis writer.
     */
    void
    registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator);

    /**
     * Set up parameters in the load balancer object (owned by the gridding
     * algorithm) if needed.  The Euler model allows non-uniform load balancing
     * to be used based on the input file parameter called
     * "use_nonuniform_workload".  The default case is to use uniform
     * load balancing (i.e., use_nonuniform_workload == false).  For
     * illustrative and testing purposes, when non-uniform load balancing is
     * turned on, a weight of one will be applied to every grid cell.  This
     * should produce an identical patch configuration to the uniform load
     * balance case.
     */
    void
    setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                      SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm);

    /**
     * Set the data on the patch interior to some initial values,
     * depending on the input parameters and numerical routines.
     * If the "initial_time" flag is false, indicating that the
     * routine is called after a regridding step, the routine does nothing.
     */
    void
    initializeDataOnPatch(SAMRAI::hier::Patch &patch,
                          const double data_time,
                          const bool initial_time);

    /**
     * Compute the stable time increment for patch using a CFL
     * condition and return the computed dt.
     */
    double
    computeStableDtOnPatch(SAMRAI::hier::Patch &patch,
                           const bool initial_time,
                           const double dt_time);

    /**
     * Compute time integral of fluxes to be used in conservative difference
     * for patch integration.  When (dim == tbox::Dimension(3)), this function calls either
     * compute3DFluxesWithCornerTransport1(), or
     * compute3DFluxesWithCornerTransport2() depending on which
     * transverse flux correction option that is specified in input.
     * The conservative difference used to update the integrated quantities
     * is implemented in the conservativeDifferenceOnPatch() routine.
     */
    void
    computeFluxesOnPatch(SAMRAI::hier::Patch &patch,
                         const double time,
                         const double dt);

    /**
     * Update linear advection solution variables by performing a conservative
     * difference with the fluxes calculated in computeFluxesOnPatch().
     */
    void
    conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch,
                                  const double time,
                                  const double dt,
                                  bool at_syncronization);

    /**
     * Tag cells for refinement using gradient detector.
     */
    void
    tagGradientDetectorCells(SAMRAI::hier::Patch &patch,
                             const double regrid_time,
                             const bool initial_error,
                             const int tag_indexindx,
                             const bool uses_richardson_extrapolation_too);

    /**
     * Tag cells for refinement using Richardson extrapolation.
     */
    void
    tagRichardsonExtrapolationCells(SAMRAI::hier::Patch &patch,
                                    const int error_level_number,
                                    const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
                                    const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse,
                                    const double regrid_time,
                                    const double deltat,
                                    const int error_coarsen_ratio,
                                    const bool initial_error,
                                    const int tag_index,
                                    const bool uses_gradient_detector_too);

    //@{
    //! @name Required implementations of HyperbolicPatchStrategy pure virtuals.

    ///
    ///  The following routines:
    ///
    ///      setPhysicalBoundaryConditions()
    ///      getRefineOpStencilWidth(),
    ///      preprocessRefine()
    ///      postprocessRefine()
    ///
    ///  are concrete implementations of functions declared in the
    ///  RefinePatchStrategy abstract base class.  Some are trivial
    ///  because this class doesn't do any pre/postprocessRefine.
    ///

    /**
     * Set the data in ghost cells corresponding to physical boundary
     * conditions.  Specific boundary conditions are determined by
     * information specified in input file and numerical routines.
     */
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch &patch,
                                       const double fill_time,
                                       const SAMRAI::hier::IntVector &
                                       ghost_width_to_fill);

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const { return SAMRAI::hier::IntVector::getZero(dim); }

    void preprocessRefine(SAMRAI::hier::Patch &fine,
                          const SAMRAI::hier::Patch &coarse,
                          const SAMRAI::hier::Box &fine_box,
                          const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    void postprocessRefine(SAMRAI::hier::Patch &fine,
                           const SAMRAI::hier::Patch &coarse,
                           const SAMRAI::hier::Box &fine_box,
                           const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    ///
    ///  The following routines:
    ///
    ///      getCoarsenOpStencilWidth(),
    ///      preprocessCoarsen()
    ///      postprocessCoarsen()
    ///
    ///  are concrete implementations of functions declared in the
    ///  CoarsenPatchStrategy abstract base class.  They are trivial
    ///  because this class doesn't do any pre/postprocessCoarsen.
    ///

    SAMRAI::hier::IntVector
    getCoarsenOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const
    {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessCoarsen(SAMRAI::hier::Patch &coarse,
                           const SAMRAI::hier::Patch &fine,
                           const SAMRAI::hier::Box &coarse_box,
                           const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    void postprocessCoarsen(SAMRAI::hier::Patch &coarse,
                            const SAMRAI::hier::Patch &fine,
                            const SAMRAI::hier::Box &coarse_box,
                            const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    //@}


    void checkUserTagData(SAMRAI::hier::Patch &patch, const int tag_index) const;

    void checkNewPatchTagData(SAMRAI::hier::Patch &patch, const int tag_index) const;


    /**
     * Register a VisIt data writer so this class will write
     * plot files that may be postprocessed with the VisIt
     * visualization tool.
     */
    void registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer);


    /**
     * Print all data members for LinAdv class.
     */
    void printClassData(std::ostream &os) const;

private:


    /*
     * The object name is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string d_object_name;

    const SAMRAI::tbox::Dimension d_dim;

    /*
     * We cache pointers to the grid geometry object to set up initial
     * data, set physical boundary conditions, and register plot
     * variables.
     */
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> d_grid_geometry;

    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer;

    /*
     * Data items used for nonuniform load balance, if used.
     */
    boost::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_workload_variable;
    int d_workload_data_id;
    bool d_use_nonuniform_workload;


};

/*
 *************************************************************************
 *
 * The constructor for SAMRAIWorkerHyperbolic class sets data members to default values,
 * creates variables that define the solution state for the linear
 * advection equation.
 *
 * After default values are set, this routine calls getFromRestart()
 * if execution from a restart file is specified.  Finally,
 * getFromInput() is called to read values from the given input
 * database (potentially overriding those found in the restart file).
 *
 *************************************************************************
 */

SAMRAIWorkerHyperbolic::SAMRAIWorkerHyperbolic(
        const std::string &object_name,
        const SAMRAI::tbox::Dimension &dim,
        boost::shared_ptr<SAMRAI::tbox::Database> input_db,
        boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geom) :
        SAMRAI::algs::HyperbolicPatchStrategy(),
        d_object_name(object_name),
        d_dim(dim),
        d_grid_geometry(grid_geom),
        d_use_nonuniform_workload(false)
{
    CHECK("This is a tag");

}

/*
 *************************************************************************
 *
 * Empty destructor for SAMRAIWorkerHyperbolic class.
 *
 *************************************************************************
 */

SAMRAIWorkerHyperbolic::~SAMRAIWorkerHyperbolic()
{
    CHECK("This is a tag");
}

/*
 *************************************************************************
 *
 * Register conserved variable (u) (i.e., solution state variable) and
 * flux variable with hyperbolic integrator that manages storage for
 * those quantities.  Also, register plot data with VisIt.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator)
{
    CHECK("This is a tag");
    TBOX_ASSERT(integrator != 0);

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
//
//    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();
//
//    if (d_visit_writer)
//    {
//        d_visit_writer->
//                registerPlotQuantity("U",
//                                     "SCALAR",
//                                     vardb->mapVariableAndContextToIndex(
//                                             d_uval, integrator->getPlotContext()));
//    }

    if (!d_visit_writer)
    {
        TBOX_WARNING(d_object_name << ": registerModelVariables()"
                                   << "\nVisIt data writer was not registered.\n"
                                   << "Consequently, no plot data will"
                                   << "\nbe written." << std::endl);
    }


}

/*
 *************************************************************************
 *
 * Set up parameters for nonuniform load balancing, if used.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                                               SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm)
{
    CHECK("This is a tag");
    const SAMRAI::hier::IntVector &zero_vec = SAMRAI::hier::IntVector::getZero(d_dim);

    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();


    if (d_use_nonuniform_workload && gridding_algorithm)
    {
        auto load_balancer = boost::dynamic_pointer_cast<SAMRAI::mesh::CascadePartitioner>(
                gridding_algorithm->getLoadBalanceStrategy());

        if (load_balancer)
        {
            d_workload_variable.reset(new SAMRAI::pdat::CellVariable<double>(d_dim, "workload_variable", 1));

            d_workload_data_id = vardb->registerVariableAndContext(d_workload_variable,
                                                                   vardb->getContext("WORKLOAD"),
                                                                   zero_vec);

            load_balancer->setWorkloadPatchDataIndex(d_workload_data_id);

        } else
        {
            TBOX_WARNING(
                    d_object_name << ": "
                                  << "  Unknown load balancer used in gridding algorithm."
                                  << "  Ignoring request for nonuniform load balancing." << std::endl);
            d_use_nonuniform_workload = false;
        }
    } else
    {
        d_use_nonuniform_workload = false;
    }

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
void SAMRAIWorkerHyperbolic::initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time,
                                                   const bool initial_time)
{
    CHECK("This is a tag");
//    if (initial_time)
//    {
//
//        auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
//        TBOX_ASSERT(pgeom);
//        const double *dx = pgeom->getDx();
//        const double *xlo = pgeom->getXLower();
//        const double *xhi = pgeom->getXUpper();
//
//
//        if (d_use_nonuniform_workload)
//        {
//            if (!patch.checkAllocated(d_workload_data_id)) { patch.allocatePatchData(d_workload_data_id); }
//
//            auto workload_data = boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>>(
//                    patch.getPatchData(d_workload_data_id));
//
//            TBOX_ASSERT(workload_data);
//
//            const SAMRAI::hier::Box &box = patch.getBox();
//            const SAMRAI::hier::BoxId &box_id = box.getBoxId();
//            const SAMRAI::hier::LocalId &local_id = box_id.getLocalId();
//
//            double id_val = local_id.getValue() % 2 ? static_cast<double>(local_id.getValue() % 10) : 0.0;
//
//            workload_data->fillAll(1.0 + id_val);
//        }
//
//    }
}

/*
 *************************************************************************
 *
 * Compute stable time increment for patch.  Return this value.
 *
 *************************************************************************
 */

double SAMRAIWorkerHyperbolic::computeStableDtOnPatch(SAMRAI::hier::Patch &patch, const bool initial_time,
                                                      const double dt_time)
{
    CHECK("This is a tag");

    auto patch_geom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
    TBOX_ASSERT(patch_geom);
    const double *dx = patch_geom->getDx();

    const SAMRAI::hier::Index ifirst = patch.getBox().lower();
    const SAMRAI::hier::Index ilast = patch.getBox().upper();

//    boost::shared_ptr<SAMRAI::pdat::CellData<double> > uval(
//            BOOST_CAST<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
//                    patch.getPatchData(d_uval, getDataContext())));
//
//    TBOX_ASSERT(uval);
//
//    SAMRAI::hier::IntVector ghost_cells(uval->getGhostCellWidth());

    double stabdt;
    if (d_dim == SAMRAI::tbox::Dimension(2))
    {
//        SAMRAI_F77_FUNC(stabledt2d, STABLEDT2D)(dx,
//                                                ifirst(0), ilast(0),
//                                                ifirst(1), ilast(1),
//                                                ghost_cells(0),
//                                                ghost_cells(1),
//                                                &d_advection_velocity[0],
//                                                stabdt);
    } else if (d_dim == SAMRAI::tbox::Dimension(3))
    {
//        SAMRAI_F77_FUNC(stabledt3d, STABLEDT3D)(dx,
//                                                ifirst(0), ilast(0),
//                                                ifirst(1), ilast(1),
//                                                ifirst(2), ilast(2),
//                                                ghost_cells(0),
//                                                ghost_cells(1),
//                                                ghost_cells(2),
//                                                &d_advection_velocity[0],
//                                                stabdt);
    } else
    {
        TBOX_ERROR("Only 2D or 3D allowed in SAMRAIWorkerHyperbolic::computeStableDtOnPatch");
        stabdt = 0;
    }

    return stabdt;
}

/*
 *************************************************************************
 *
 * Compute time integral of numerical fluxes for finite difference
 * at each cell face on patch.  When d_dim == SAMRAI::tbox::Dimension(3)), there are two options
 * for the transverse flux correction.  Otherwise, there is only one.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::computeFluxesOnPatch(
        SAMRAI::hier::Patch &patch,
        const double time,
        const double dt)
{
    CHECK("This is a tag");


}


/*
 *************************************************************************
 *
 * Update solution variables by performing a conservative
 * difference with the fluxes calculated in computeFluxesOnPatch().
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::conservativeDifferenceOnPatch(
        SAMRAI::hier::Patch &patch,
        const double time,
        const double dt,
        bool at_syncronization)
{
    CHECK("This is a tag");

}


/*
 *************************************************************************
 *
 * Set the data in ghost cells corresponding to physical boundary
 * conditions.  Note that boundary geometry configuration information
 * (i.e., faces, edges, and nodes) is obtained from the patch geometry
 * object owned by the patch.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::setPhysicalBoundaryConditions(
        SAMRAI::hier::Patch &patch,
        const double fill_time,
        const SAMRAI::hier::IntVector &ghost_width_to_fill)
{
    CHECK("This is a tag");

}

/*
 *************************************************************************
 *
 * Tag cells for refinement using Richardson extrapolation.  Criteria
 * defined in input.
 *
 *************************************************************************
 */
void SAMRAIWorkerHyperbolic::tagRichardsonExtrapolationCells(
        SAMRAI::hier::Patch &patch,
        const int error_level_number,
        const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
        const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse,
        const double regrid_time,
        const double deltat,
        const int error_coarsen_ratio,
        const bool initial_error,
        const int tag_index,
        const bool uses_gradient_detector_too)
{
    CHECK("This is a tag");


}

/*
 *************************************************************************
 *
 * Tag cells for refinement using gradient detector.  Tagging criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::tagGradientDetectorCells(
        SAMRAI::hier::Patch &patch,
        const double regrid_time,
        const bool initial_error,
        const int tag_indx,
        const bool uses_richardson_extrapolation_too)
{
    CHECK("This is a tag");


}

/*
 *************************************************************************
 *
 * Register VisIt data writer to write data to plot files that may
 * be postprocessed by the VisIt tool.
 *
 *************************************************************************
 */


void SAMRAIWorkerHyperbolic::registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer)
{
    CHECK("This is a tag");

    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}


/*
 *************************************************************************
 *
 * Write SAMRAIWorkerHyperbolic object state to specified stream.
 *
 *************************************************************************
 */

void SAMRAIWorkerHyperbolic::printClassData(std::ostream &os) const
{
    CHECK("This is a tag");

    os << "\nSAMRAIWorkerHyperbolic::printClassData..." << std::endl;
    os << "SAMRAIWorkerHyperbolic: this = " << (SAMRAIWorkerHyperbolic *) this << std::endl;
    os << "d_object_name = " << d_object_name << std::endl;
    os << "d_grid_geometry = " << d_grid_geometry.get() << std::endl;
    os << std::endl;
}


void
SAMRAIWorkerHyperbolic::checkUserTagData(SAMRAI::hier::Patch &patch, const int tag_index) const
{
    CHECK("This is a tag");

    auto tags = boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int> >(patch.getPatchData(tag_index));
    TBOX_ASSERT(tags);
}

void
SAMRAIWorkerHyperbolic::checkNewPatchTagData(SAMRAI::hier::Patch &patch, const int tag_index) const
{
    CHECK("This is a tag");

    auto tags = boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int>>
            (patch.getPatchData(tag_index));
    TBOX_ASSERT(tags);
}


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

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;

    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

    boost::shared_ptr<SAMRAI::mesh::StandardTagAndInitialize> error_detector;

    boost::shared_ptr<SAMRAI::mesh::BergerRigoutsos> box_generator;

    boost::shared_ptr<SAMRAI::mesh::CascadePartitioner> load_balancer;

    boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithm> gridding_algorithm;

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

    patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>(
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

    error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize",
            hyp_level_integrator.get(),
            input_db->getDatabase("StandardTagAndInitialize"));


    //---------------------------------

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
    //---------------------------------
    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
            "TimeRefinementIntegrator",
            input_db->getDatabase("TimeRefinementIntegrator"),
            patch_hierarchy,
            hyp_level_integrator,
            gridding_algorithm);

    /*
     * After creating all objects and initializing their state, we
     * print the input database and variable database contents
     * to the log file.
     */

    SAMRAI::tbox::plog << "\nCheck input data and variables before simulation:" << std::endl;
    SAMRAI::tbox::plog << "Input database..." << std::endl;
    input_db->printClassData(SAMRAI::tbox::plog);
    SAMRAI::tbox::plog << "\nVariable database..." << std::endl;
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(SAMRAI::tbox::plog);

    SAMRAI::tbox::plog << "\nCheck Linear Advection data... " << std::endl;
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
    gridding_algorithm.reset();
    load_balancer.reset();
    box_generator.reset();
    error_detector.reset();
    hyp_level_integrator.reset();
    patch_worker.reset();

    patch_hierarchy.reset();
    grid_geometry.reset();

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