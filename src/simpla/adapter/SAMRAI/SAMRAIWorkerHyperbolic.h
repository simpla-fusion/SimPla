/*************************************************************************
 *
 * This file is part of the SAMRAI distribution.  For full copyright
 * information, see COPYRIGHT and COPYING.LESSER.
 *
 * Copyright:     (c) 1997-2015 Lawrence Livermore National Security, LLC
 * Description:   Numerical routines for single patch in linear advection ex.
 *
 ************************************************************************/

#ifndef included_SAMRAIWorkerHyperbolicXD
#define included_SAMRAIWorkerHyperbolicXD


#include <SAMRAI/SAMRAI_config.h>

#include <SAMRAI/appu/BoundaryUtilityStrategy.h>
#include <SAMRAI/hier/Box.h>
#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/mesh/GriddingAlgorithm.h>
#include <SAMRAI/algs/HyperbolicLevelIntegrator.h>
#include <SAMRAI/algs/HyperbolicPatchStrategy.h>
#include <SAMRAI/hier/IntVector.h>
#include <SAMRAI/hier/Patch.h>
#include <SAMRAI/tbox/Serializable.h>
#include <SAMRAI/hier/VariableContext.h>
#include <SAMRAI/appu/VisItDataWriter.h>

#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>

namespace simpla
{


using namespace SAMRAI;

class SAMRAIWorkerHyperbolic :
        public tbox::Serializable,
        public algs::HyperbolicPatchStrategy,
        public appu::BoundaryUtilityStrategy
{
public:

    SAMRAIWorkerHyperbolic(
            const std::string &object_name,
            const tbox::Dimension &dim,
            boost::shared_ptr<tbox::Database> input_db,
            boost::shared_ptr<geom::CartesianGridGeometry> grid_geom);

    /**
     * The destructor for SAMRAIWorkerHyperbolic does nothing.
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
     * Register SAMRAIWorkerHyperbolic model variables with algs::HyperbolicLevelIntegrator
     * according to variable registration function provided by the integrator.
     * In other words, variables are registered according to their role
     * in the integration process (e.g., time-dependent, flux, etc.).
     * This routine also registers variables for plotting with the
     * Vis writer.
     */
    void
    registerModelVariables(
            algs::HyperbolicLevelIntegrator *integrator);

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
    setupLoadBalancer(
            algs::HyperbolicLevelIntegrator *integrator,
            SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm);

    /**
     * Set the data on the patch interior to some initial values,
     * depending on the input parameters and numerical routines.
     * If the "initial_time" flag is false, indicating that the
     * routine is called after a regridding step, the routine does nothing.
     */
    void
    initializeDataOnPatch(
            hier::Patch &patch,
            const double data_time,
            const bool initial_time);

    /**
     * Compute the stable time increment for patch using a CFL
     * condition and return the computed dt.
     */
    double
    computeStableDtOnPatch(
            hier::Patch &patch,
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
    computeFluxesOnPatch(
            hier::Patch &patch,
            const double time,
            const double dt);

    /**
     * Update linear advection solution variables by performing a conservative
     * difference with the fluxes calculated in computeFluxesOnPatch().
     */
    void
    conservativeDifferenceOnPatch(
            hier::Patch &patch,
            const double time,
            const double dt,
            bool at_syncronization);

    /**
     * Tag cells for refinement using gradient detector.
     */
    void
    tagGradientDetectorCells(
            hier::Patch &patch,
            const double regrid_time,
            const bool initial_error,
            const int tag_indexindx,
            const bool uses_richardson_extrapolation_too);

    /**
     * Tag cells for refinement using Richardson extrapolation.
     */
    void
    tagRichardsonExtrapolationCells(
            hier::Patch &patch,
            const int error_level_number,
            const boost::shared_ptr<hier::VariableContext> &coarsened_fine,
            const boost::shared_ptr<hier::VariableContext> &advanced_coarse,
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
    void
    setPhysicalBoundaryConditions(
            hier::Patch &patch,
            const double fill_time,
            const hier::IntVector &
            ghost_width_to_fill);

    hier::IntVector
    getRefineOpStencilWidth(const tbox::Dimension &dim) const
    {
        return hier::IntVector::getZero(dim);
    }

    void
    preprocessRefine(
            hier::Patch &fine,
            const hier::Patch &coarse,
            const hier::Box &fine_box,
            const hier::IntVector &ratio)
    {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    void
    postprocessRefine(
            hier::Patch &fine,
            const hier::Patch &coarse,
            const hier::Box &fine_box,
            const hier::IntVector &ratio)
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

    hier::IntVector
    getCoarsenOpStencilWidth(const tbox::Dimension &dim) const
    {
        return hier::IntVector::getZero(dim);
    }

    void
    preprocessCoarsen(
            hier::Patch &coarse,
            const hier::Patch &fine,
            const hier::Box &coarse_box,
            const hier::IntVector &ratio)
    {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    void
    postprocessCoarsen(
            hier::Patch &coarse,
            const hier::Patch &fine,
            const hier::Box &coarse_box,
            const hier::IntVector &ratio)
    {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    //@}

    /**
     * Write state of SAMRAIWorkerHyperbolic object to the given database for restart.
     *
     * This routine is a concrete implementation of the function
     * declared in the tbox::Serializable abstract base class.
     */
    void
    putToRestart(
            const boost::shared_ptr<tbox::Database> &restart_db) const;

    /**
     * This routine is a concrete implementation of the virtual function
     * in the base class BoundaryUtilityStrategy.  It reads DIRICHLET
     * boundary state values from the given database with the
     * given name string idenifier.  The integer location index
     * indicates the face (in 3D) or edge (in 2D) to which the boundary
     * condition applies.
     */
    void
    readDirichletBoundaryDataEntry(
            const boost::shared_ptr<tbox::Database> &db,
            std::string &db_name,
            int bdry_location_index);

    /**
     * This routine is a concrete implementation of the virtual function
     * in the base class BoundaryUtilityStrategy.  It is a blank implementation
     * for the purposes of this class.
     */
    void
    readNeumannBoundaryDataEntry(
            const boost::shared_ptr<tbox::Database> &db,
            std::string &db_name,
            int bdry_location_index);

    void
    checkUserTagData(
            hier::Patch &patch,
            const int tag_index) const;

    void
    checkNewPatchTagData(
            hier::Patch &patch,
            const int tag_index) const;


    /**
     * Register a VisIt data writer so this class will write
     * plot files that may be postprocessed with the VisIt
     * visualization tool.
     */
    void registerVisItDataWriter(boost::shared_ptr<appu::VisItDataWriter> viz_writer);


    /**
     * Reset physical boundary values in special cases, such as when
     * using symmetric (i.e., reflective) boundary conditions.
     */
    void boundaryReset(
            hier::Patch &patch,
            pdat::FaceData<double> &traced_left,
            pdat::FaceData<double> &traced_right) const;

    /**
     * Print all data members for SAMRAIWorkerHyperbolic class.
     */
    void printClassData(std::ostream &os) const;

private:
    /*
     * These private member functions read data from input and restart.
     * When beginning a run from a restart file, all data members are read
     * from the restart file.  If the boolean flag is true when reading
     * from input, some restart values may be overridden by those in the
     * input file.
     *
     * An assertion results if the database pointer is null.
     */
    void
    getFromInput(
            boost::shared_ptr<tbox::Database> input_db,
            bool is_from_restart);

    void
    getFromRestart();

    void
    readStateDataEntry(
            boost::shared_ptr<tbox::Database> db,
            const std::string &db_name,
            int array_indx,
            std::vector<double> &uval);

    /*
     * Private member function to check correctness of boundary data.
     */
    void
    checkBoundaryData(
            int btype,
            const hier::Patch &patch,
            const hier::IntVector &ghost_width_to_fill,
            const std::vector<int> &scalar_bconds) const;

    /*
     * Three-dimensional flux computation routines corresponding to
     * either of the two transverse flux correction options.  These
     * routines are called from the computeFluxesOnPatch() function.
     */
    void
    compute3DFluxesWithCornerTransport1(
            hier::Patch &patch,
            const double dt);

    void
    compute3DFluxesWithCornerTransport2(
            hier::Patch &patch,
            const double dt);

    /*
     * The object name is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string d_object_name;

    const tbox::Dimension d_dim;

    /*
     * We cache pointers to the grid geometry object to set up initial
     * data, set physical boundary conditions, and register plot
     * variables.
     */
    boost::shared_ptr<geom::CartesianGridGeometry> d_grid_geometry;

    boost::shared_ptr<appu::VisItDataWriter> d_visit_writer;

    /*
     * Data items used for nonuniform load balance, if used.
     */
    boost::shared_ptr<pdat::CellVariable<double> > d_workload_variable;
    int d_workload_data_id;
    bool d_use_nonuniform_workload;

    /**
     * boost::shared_ptr to state variable vector - [u]
     */
    boost::shared_ptr<pdat::CellVariable<double> > d_uval;

    /**
     * boost::shared_ptr to flux variable vector  - [F]
     */
    boost::shared_ptr<pdat::FaceVariable<double> > d_flux;

    /**
     * linear advection velocity vector
     */
    std::vector<double> d_advection_velocity;

    /*
     *  Parameters for numerical method:
     *
     *    d_godunov_order ....... order of Godunov slopes (1, 2, or 4)
     *
     *    d_corner_transport .... type of finite difference approximation
     *                            for 3d transverse flux correction
     *
     *    d_nghosts ............. number of ghost cells for cell-centered
     *                            and face/side-centered variables
     *
     *    d_fluxghosts .......... number of ghost cells for fluxes
     *
     */
    int d_godunov_order;
    std::string d_corner_transport;
    hier::IntVector d_nghosts;
    hier::IntVector d_fluxghosts;

    /*
     * Indicator for problem type and initial conditions
     */
    std::string d_data_problem;
    int d_data_problem_int;

    /*
     * Input for SPHERE problem
     */
    double d_radius;
    std::vector<double> d_center;
    double d_uval_inside;
    double d_uval_outside;

    /*
     * Input for FRONT problem
     */
    int d_number_of_intervals;
    std::vector<double> d_front_position;
    std::vector<double> d_interval_uval;

    /*
     * Boundary condition cases and boundary values.
     * Options are: FLOW, REFLECT, DIRICHLET
     * and variants for nodes and edges.
     *
     * Input file values are read into these arrays.
     */
    std::vector<int> d_scalar_bdry_edge_conds;
    std::vector<int> d_scalar_bdry_node_conds;
    std::vector<int> d_scalar_bdry_face_conds; // only for (dim == tbox::Dimension(3))

    /*
     * Boundary condition cases for scalar and vector (i.e., depth > 1)
     * variables.  These are post-processed input values and are passed
     * to the boundary routines.
     */
    std::vector<int> d_node_bdry_edge; // only for (dim == tbox::Dimension(2))
    std::vector<int> d_edge_bdry_face; // only for (dim == tbox::Dimension(3))
    std::vector<int> d_node_bdry_face; // only for (dim == tbox::Dimension(3))

    /*
     * Vectors of face (3d) or edge (2d) boundary values for DIRICHLET case.
     */
    std::vector<double> d_bdry_edge_uval; // only for (dim == tbox::Dimension(2))
    std::vector<double> d_bdry_face_uval; // only for (dim == tbox::Dimension(3))

    /*
     * Input for Sine problem initialization
     */
    double d_amplitude;
    std::vector<double> d_frequency;

    /*
     * Refinement criteria parameters for gradient detector and
     * Richardson extrapolation.
     */
    std::vector<std::string> d_refinement_criteria;
    std::vector<double> d_dev_tol;
    std::vector<double> d_dev;
    std::vector<double> d_dev_time_max;
    std::vector<double> d_dev_time_min;
    std::vector<double> d_grad_tol;
    std::vector<double> d_grad_time_max;
    std::vector<double> d_grad_time_min;
    std::vector<double> d_shock_onset;
    std::vector<double> d_shock_tol;
    std::vector<double> d_shock_time_max;
    std::vector<double> d_shock_time_min;
    std::vector<double> d_rich_tol;
    std::vector<double> d_rich_time_max;
    std::vector<double> d_rich_time_min;

};
}//namespace simpla{
#endif
