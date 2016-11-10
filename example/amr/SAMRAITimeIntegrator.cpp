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
#include <SAMRAI/pdat/SideVariable.h>

namespace simpla
{
struct SAMRAIWorker;

struct SAMRAITimeIntegrator;

std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name, std::shared_ptr<mesh::Worker> const &w)
{
    return std::dynamic_pointer_cast<simulation::TimeIntegrator>(std::make_shared<SAMRAITimeIntegrator>(name, w));
}

/*********************************************************************************************************************/

//integer constants for boundary conditions
#define CHECK_BDRY_DATA (0)

#include <SAMRAI/appu/CartesianBoundaryDefines.h>

//integer constant for debugging improperly set boundary dat
#define BOGUS_BDRY_DATA (-9999)

// Number of ghosts cells used for each variable quantity
#define CELLG (4)
#define FACEG (4)
#define FLUXG (1)

// defines for initialization
#define PIECEWISE_CONSTANT_X (10)
#define PIECEWISE_CONSTANT_Y (11)
#define PIECEWISE_CONSTANT_Z (12)
#define SINE_CONSTANT_X (20)
#define SINE_CONSTANT_Y (21)
#define SINE_CONSTANT_Z (22)
#define SPHERE (40)

// defines for Riemann solver used in Godunov flux calculation
#define APPROX_RIEM_SOLVE (20)   // Colella-Glaz approx Riemann solver
#define EXACT_RIEM_SOLVE (21)    // Exact Riemann solver
#define HLLC_RIEM_SOLVE (22)     // Harten, Lax, van Leer approx Riemann solver

// defines for cell tagging routines
#define RICHARDSON_NEWLY_TAGGED (-10)
#define RICHARDSON_ALREADY_TAGGED (-11)
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif

// Version of LinAdv restart file data
#define LINADV_VERSION (3)

class SAMRAIWorker :
        public SAMRAI::algs::HyperbolicPatchStrategy,
        public SAMRAI::appu::BoundaryUtilityStrategy // for Boundary
{

public:

    SAMRAIWorker(std::shared_ptr<mesh::Worker> const &w,
                 const SAMRAI::tbox::Dimension &dim,
                 boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geom);

    /**
     * The destructor for SAMRAIWorkerHyperbolic does nothing.
     */
    ~SAMRAIWorker();



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

    void registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator);


    void setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                           SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm);

    void initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time, const bool initial_time);


    double computeStableDtOnPatch(SAMRAI::hier::Patch &patch, const bool initial_time, const double dt_time);


    void computeFluxesOnPatch(SAMRAI::hier::Patch &patch, const double time, const double dt);

    /**
     * Update linear advection solution variables by performing a conservative
     * difference with the fluxes calculated in computeFluxesOnPatch().
     */
    void conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, const double time, const double dt,
                                       bool at_syncronization);

    /**
     * Tag cells for refinement using gradient detector.
     */
    void tagGradientDetectorCells(
            SAMRAI::hier::Patch &patch,
            const double regrid_time,
            const bool initial_error,
            const int tag_indexindx,
            const bool uses_richardson_extrapolation_too);

    /**
     * Tag cells for refinement using Richardson extrapolation.
     */
    void tagRichardsonExtrapolationCells(
            SAMRAI::hier::Patch &patch,
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
    void setPhysicalBoundaryConditions(
            SAMRAI::hier::Patch &patch,
            const double fill_time,
            const SAMRAI::hier::IntVector &
            ghost_width_to_fill);

    SAMRAI::hier::IntVector
    getRefineOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const { return SAMRAI::hier::IntVector::getZero(dim); }

    void
    preprocessRefine(
            SAMRAI::hier::Patch &fine,
            const SAMRAI::hier::Patch &coarse,
            const SAMRAI::hier::Box &fine_box,
            const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    void
    postprocessRefine(
            SAMRAI::hier::Patch &fine,
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

    void
    preprocessCoarsen(
            SAMRAI::hier::Patch &coarse,
            const SAMRAI::hier::Patch &fine,
            const SAMRAI::hier::Box &coarse_box,
            const SAMRAI::hier::IntVector &ratio)
    {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    void
    postprocessCoarsen(
            SAMRAI::hier::Patch &coarse,
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
//
//    /**
//     * Write state of SAMRAIWorker object to the given database for restart.
//     *
//     * This routine is a concrete implementation of the function
//     * declared in the tbox::Serializable abstract base class.
//     */
//    void putToRestart(const boost::shared_ptr<SAMRAI::tbox::Database> &restart_db) const;
//
    /**
     * This routine is a concrete implementation of the virtual function
     * in the base class BoundaryUtilityStrategy.  It reads DIRICHLET
     * boundary state values from the given database with the
     * given name string idenifier.  The integer location index
     * indicates the face (in 3D) or edge (in 2D) to which the boundary
     * condition applies.
     */
    void readDirichletBoundaryDataEntry(const boost::shared_ptr<SAMRAI::tbox::Database> &db,
                                        std::string &db_name, int bdry_location_index);

    /**
     * This routine is a concrete implementation of the virtual function
     * in the base class BoundaryUtilityStrategy.  It is a blank implementation
     * for the purposes of this class.
     */
    void readNeumannBoundaryDataEntry(const boost::shared_ptr<SAMRAI::tbox::Database> &db,
                                      std::string &db_name, int bdry_location_index);

//    void checkUserTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const;
//
//    void checkNewPatchTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const;


    /**
     * Register a VisIt data writer so this class will write
     * plot files that may be postprocessed with the VisIt
     * visualization tool.
     */
    void registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer);

//
//    /**
//     * Reset physical boundary values in special cases, such as when
//     * using symmetric (i.e., reflective) boundary conditions.
//     */
    void boundaryReset(SAMRAI::hier::Patch &patch, SAMRAI::pdat::FaceData<double> &traced_left,
                       SAMRAI::pdat::FaceData<double> &traced_right) const;

    /**
     * Print all data members for SAMRAIWorkerHyperbolic class.
     */
    void printClassData(std::ostream &os) const;

private:
    void move_to(mesh::Worker *w, SAMRAI::hier::Patch &patch);

private:
    std::shared_ptr<mesh::Worker> m_worker_;
    std::shared_ptr<mesh::Atlas> m_atlas_;
    /*
     * The object name is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string m_name_;

    const SAMRAI::tbox::Dimension d_dim;

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
    bool d_use_nonuniform_workload;


    std::map<id_type, boost::shared_ptr<SAMRAI::hier::Variable>> m_samrai_variables_;

    /**
     * boost::shared_ptr to state variable vector - [u]
     */
    boost::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_uval;

    /**
     * boost::shared_ptr to flux variable vector  - [F]
     */
    boost::shared_ptr<SAMRAI::pdat::FaceVariable<double>> d_flux;

    /**
     * linear advection velocity vector
     */
    std::vector<double> d_advection_velocity;


    int d_godunov_order;
    std::string d_corner_transport;
    SAMRAI::hier::IntVector d_nghosts;
    SAMRAI::hier::IntVector d_fluxghosts;

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

SAMRAIWorker::SAMRAIWorker(
        std::shared_ptr<mesh::Worker> const &w,
        const SAMRAI::tbox::Dimension &dim,
        boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geom) :
        SAMRAI::algs::HyperbolicPatchStrategy(),
        m_worker_(w),
        m_name_(w != nullptr ? w->name() : ""),
        d_dim(dim),
        d_grid_geometry(grid_geom),
        d_use_nonuniform_workload(false),
        d_uval(new SAMRAI::pdat::CellVariable<double>(dim, "uval", 1)),
        d_flux(new SAMRAI::pdat::FaceVariable<double>(dim, "flux", 1)),
//        d_advection_velocity(dim.getValue()),
//        d_godunov_order(1),
        d_corner_transport("CORNER_TRANSPORT_1"),
        d_nghosts(dim, CELLG),
        d_fluxghosts(dim, FLUXG),
        d_data_problem("SPHERE"),
        d_data_problem_int(SAMRAI::tbox::MathUtilities<int>::getMax()),
        d_radius(SAMRAI::tbox::MathUtilities<double>::getSignalingNaN()),
        d_center(dim.getValue()),
//        d_uval_inside(SAMRAI::tbox::MathUtilities<double>::getSignalingNaN()),
//        d_uval_outside(SAMRAI::tbox::MathUtilities<double>::getSignalingNaN()),
        d_number_of_intervals(0),
        d_amplitude(0.),
        d_frequency(dim.getValue())
{
    TBOX_ASSERT(grid_geom);
    TBOX_ASSERT(CELLG == FACEG);

    // SPHERE problem...
    SAMRAI::tbox::MathUtilities<double>::setVectorToSignalingNaN(d_center);

    // SINE problem
    for (int k = 0; k < d_dim.getValue(); ++k) d_frequency[k] = 0.;

//    /*
//     * Defaults for boundary conditions. Set to bogus values
//     * for error checking.
//     */
//
//    if (d_dim == SAMRAI::tbox::Dimension(2))
//    {
//        d_scalar_bdry_edge_conds.resize(NUM_2D_EDGES);
//        for (int ei = 0; ei < NUM_2D_EDGES; ++ei)
//        {
//            d_scalar_bdry_edge_conds[ei] = BOGUS_BDRY_DATA;
//        }
//
//        d_scalar_bdry_node_conds.resize(NUM_2D_NODES);
//        d_node_bdry_edge.resize(NUM_2D_NODES);
//
//        for (int ni = 0; ni < NUM_2D_NODES; ++ni)
//        {
//            d_scalar_bdry_node_conds[ni] = BOGUS_BDRY_DATA;
//            d_node_bdry_edge[ni] = BOGUS_BDRY_DATA;
//        }
//
//        d_bdry_edge_uval.resize(NUM_2D_EDGES);
//        SAMRAI::tbox::MathUtilities<double>::setVectorToSignalingNaN(d_bdry_edge_uval);
//    }
//    if (d_dim == SAMRAI::tbox::Dimension(3))
//    {
//        d_scalar_bdry_face_conds.resize(NUM_3D_FACES);
//        for (int fi = 0; fi < NUM_3D_FACES; ++fi)
//        {
//            d_scalar_bdry_face_conds[fi] = BOGUS_BDRY_DATA;
//        }
//
//        d_scalar_bdry_edge_conds.resize(NUM_3D_EDGES);
//        d_edge_bdry_face.resize(NUM_3D_EDGES);
//        for (int ei = 0; ei < NUM_3D_EDGES; ++ei)
//        {
//            d_scalar_bdry_edge_conds[ei] = BOGUS_BDRY_DATA;
//            d_edge_bdry_face[ei] = BOGUS_BDRY_DATA;
//        }
//
//        d_scalar_bdry_node_conds.resize(NUM_3D_NODES);
//        d_node_bdry_face.resize(NUM_3D_NODES);
//
//        for (int ni = 0; ni < NUM_3D_NODES; ++ni)
//        {
//            d_scalar_bdry_node_conds[ni] = BOGUS_BDRY_DATA;
//            d_node_bdry_face[ni] = BOGUS_BDRY_DATA;
//        }
//
//        d_bdry_face_uval.resize(NUM_3D_FACES);
//        SAMRAI::tbox::MathUtilities<double>::setVectorToSignalingNaN(d_bdry_face_uval);
//    }
//    /*
//     * Initialize object with data read from given input/restart databases.
//     */
//
//    auto input_db = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("LinAdv");
//    double advection_velocity[3] = {2.0e0, 1.0e0, 1.0e0};
//    input_db->putDoubleArray("advection_velocity", advection_velocity, 3);
//    input_db->putInteger("godunov_order", 2);
//    input_db->putString("corner_transport", "CORNER_TRANSPORT_1");
//    input_db->putString("data_problem", "SPHERE");
//
//    auto Initial_data_db = input_db->putDatabase("Initial_data");
//    double center[3] = {5.5, 5.5, 5.5};
//    Initial_data_db->putDouble("radius", 2.9);
//    Initial_data_db->putDoubleArray("center", center, 3);
//    Initial_data_db->putDouble("uval_inside", 80.0);
//    Initial_data_db->putDouble("uval_outside", 5.0);
//
//    auto Boundary_data_db = input_db->putDatabase("Boundary_data");
//    Boundary_data_db->putDatabase("boundary_face_xlo")->putString("boundary_condition", "FLOW");
//    Boundary_data_db->putDatabase("boundary_face_xhi")->putString("boundary_condition", "FLOW");
//    Boundary_data_db->putDatabase("boundary_face_ylo")->putString("boundary_condition", "FLOW");
//    Boundary_data_db->putDatabase("boundary_face_yhi")->putString("boundary_condition", "FLOW");
//    Boundary_data_db->putDatabase("boundary_face_zlo")->putString("boundary_condition", "FLOW");
//    Boundary_data_db->putDatabase("boundary_face_zhi")->putString("boundary_condition", "FLOW");
//
//
//    Boundary_data_db->putDatabase("boundary_edge_ylo_zlo")->putString("boundary_condition", "ZFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_yhi_zlo")->putString("boundary_condition", "ZFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_ylo_zhi")->putString("boundary_condition", "ZFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_yhi_zhi")->putString("boundary_condition", "ZFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xlo_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xlo_zhi")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xhi_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xhi_zhi")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xlo_ylo")->putString("boundary_condition", "YFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xhi_ylo")->putString("boundary_condition", "YFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xlo_yhi")->putString("boundary_condition", "YFLOW");
//    Boundary_data_db->putDatabase("boundary_edge_xhi_yhi")->putString("boundary_condition", "YFLOW");
//
//
//    Boundary_data_db->putDatabase("boundary_node_xlo_ylo_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xhi_ylo_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xlo_yhi_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xhi_yhi_zlo")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xlo_ylo_zhi")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xhi_ylo_zhi")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xlo_yhi_zhi")->putString("boundary_condition", "XFLOW");
//    Boundary_data_db->putDatabase("boundary_node_xhi_yhi_zhi")->putString("boundary_condition", "XFLOW");
//
//
//
//
//    /*
//     * Set problem data to values read from input/restart.
//     */
//
//    if (d_data_problem == "PIECEWISE_CONSTANT_X") { d_data_problem_int = PIECEWISE_CONSTANT_X; }
//    else if (d_data_problem == "PIECEWISE_CONSTANT_Y") { d_data_problem_int = PIECEWISE_CONSTANT_Y; }
//    else if (d_data_problem == "PIECEWISE_CONSTANT_Z") { d_data_problem_int = PIECEWISE_CONSTANT_Z; }
//    else if (d_data_problem == "SINE_CONSTANT_X") { d_data_problem_int = SINE_CONSTANT_X; }
//    else if (d_data_problem == "SINE_CONSTANT_Y") { d_data_problem_int = SINE_CONSTANT_Y; }
//    else if (d_data_problem == "SINE_CONSTANT_Z") { d_data_problem_int = SINE_CONSTANT_Z; }
//    else if (d_data_problem == "SPHERE") { d_data_problem_int = SPHERE; }
//    else
//    {
//        TBOX_ERROR(
//                m_name_ << ": "
//                              << "Unknown d_data_problem string = "
//                              << d_data_problem
//                              << " encountered in constructor" << std::endl);
//    }
//
//    /*
//     * Postprocess boundary data from input/restart values.  Note: scalar
//     * quantity in this problem cannot have reflective boundary conditions
//     * so we reset them to FLOW.
//     */
//    if (d_dim == SAMRAI::tbox::Dimension(2))
//    {
//        for (int i = 0; i < NUM_2D_EDGES; ++i)
//        {
//            if (d_scalar_bdry_edge_conds[i] == BdryCond::REFLECT) { d_scalar_bdry_edge_conds[i] = BdryCond::FLOW; }
//        }
//
//        for (int i = 0; i < NUM_2D_NODES; ++i)
//        {
//            if (d_scalar_bdry_node_conds[i] == BdryCond::XREFLECT) { d_scalar_bdry_node_conds[i] = BdryCond::XFLOW; }
//            if (d_scalar_bdry_node_conds[i] == BdryCond::YREFLECT) { d_scalar_bdry_node_conds[i] = BdryCond::YFLOW; }
//
//            if (d_scalar_bdry_node_conds[i] != BOGUS_BDRY_DATA)
//            {
//                d_node_bdry_edge[i] =
//                        SAMRAI::appu::CartesianBoundaryUtilities2::getEdgeLocationForNodeBdry(
//                                i, d_scalar_bdry_node_conds[i]);
//            }
//        }
//    }
//    if (d_dim == SAMRAI::tbox::Dimension(3))
//    {
//        for (int i = 0; i < NUM_3D_FACES; ++i)
//        {
//            if (d_scalar_bdry_face_conds[i] == BdryCond::REFLECT) { d_scalar_bdry_face_conds[i] = BdryCond::FLOW; }
//        }
//
//        for (int i = 0; i < NUM_3D_EDGES; ++i)
//        {
//            if (d_scalar_bdry_edge_conds[i] == BdryCond::XREFLECT) { d_scalar_bdry_edge_conds[i] = BdryCond::XFLOW; }
//            if (d_scalar_bdry_edge_conds[i] == BdryCond::YREFLECT) { d_scalar_bdry_edge_conds[i] = BdryCond::YFLOW; }
//            if (d_scalar_bdry_edge_conds[i] == BdryCond::ZREFLECT) { d_scalar_bdry_edge_conds[i] = BdryCond::ZFLOW; }
//
//            if (d_scalar_bdry_edge_conds[i] != BOGUS_BDRY_DATA)
//            {
//                d_edge_bdry_face[i] = SAMRAI::appu::CartesianBoundaryUtilities3::getFaceLocationForEdgeBdry(i,
//                                                                                                            d_scalar_bdry_edge_conds[i]);
//            }
//        }
//
//        for (int i = 0; i < NUM_3D_NODES; ++i)
//        {
//            if (d_scalar_bdry_node_conds[i] == BdryCond::XREFLECT) { d_scalar_bdry_node_conds[i] = BdryCond::XFLOW; }
//            if (d_scalar_bdry_node_conds[i] == BdryCond::REFLECT) { d_scalar_bdry_node_conds[i] = BdryCond::YFLOW; }
//            if (d_scalar_bdry_node_conds[i] == BdryCond::ZREFLECT) { d_scalar_bdry_node_conds[i] = BdryCond::ZFLOW; }
//
//            if (d_scalar_bdry_node_conds[i] != BOGUS_BDRY_DATA)
//            {
//                d_node_bdry_face[i] = SAMRAI::appu::CartesianBoundaryUtilities3::getFaceLocationForNodeBdry(i,
//                                                                                                            d_scalar_bdry_node_conds[i]);
//            }
//        }
//
//    }
//
//    SAMRAI_F77_FUNC(stufprobc, STUFPROBC)(PIECEWISE_CONSTANT_X, PIECEWISE_CONSTANT_Y,
//                                          PIECEWISE_CONSTANT_Z,
//                                          SINE_CONSTANT_X, SINE_CONSTANT_Y, SINE_CONSTANT_Z, SPHERE,
//                                          CELLG, FACEG, FLUXG);

}

/*
 *************************************************************************
 *
 * Empty destructor for SAMRAIWorker class.
 *
 *************************************************************************
 */

SAMRAIWorker::~SAMRAIWorker()
{
}

namespace detail
{
static const char visit_variable_type[3][10] = {"SCALAR", "VECTOR", "TENSOR"};


template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Node(mesh::Attribute *item,
                       int ndims,
                       SAMRAI::hier::IntVector const &d_nghosts,
                       SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &d_grid_geometry,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(ndims);

    auto var = boost::make_shared<SAMRAI::pdat::NodeVariable<T>>(d_dim, item->name(), item->value_size());
    integrator->registerVariable(var, d_nghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                 d_grid_geometry,
                                 "",
                                 "LINEAR_REFINE");
    d_visit_writer->registerPlotQuantity(item->name(), visit_variable_type[item->value_rank()],
                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}

template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Edge(mesh::Attribute *item,
                       int ndims,
                       SAMRAI::hier::IntVector const &d_nghosts,
                       SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &d_grid_geometry,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(ndims);

    auto var = boost::make_shared<SAMRAI::pdat::EdgeVariable<T>>(d_dim, item->name(), item->value_size());
    integrator->registerVariable(var, d_nghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                 d_grid_geometry,
                                 "CONSERVATIVE_COARSEN",
                                 "CONSERVATIVE_LINEAR_REFINE");

    /*** FIXME:
      *  1. SAMRAI Visit Writer only support NODE and CELL variable (double,float ,int)
      *  2. SAMRAI   SAMRAI::algs::HyperbolicLevelIntegrator->registerVariable only support double
     **/
//    d_visit_writer->registerPlotQuantity(item->name(),  visit_variable_type[item->value_rank()],
//                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()) );


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}

template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Face(mesh::Attribute *item,
                       int ndims,
                       SAMRAI::hier::IntVector const &d_nghosts,
                       SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &d_grid_geometry,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(ndims);

    auto var = boost::make_shared<SAMRAI::pdat::FaceVariable<T>>(d_dim, item->name(), item->value_size());
    integrator->registerVariable(var, d_nghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                 d_grid_geometry,
                                 "CONSERVATIVE_COARSEN",
                                 "CONSERVATIVE_LINEAR_REFINE");

//    d_visit_writer->registerPlotQuantity(item->name(), visit_variable_type[item->value_rank()],
//                                         vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));


    return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(var);
}


template<typename T> boost::shared_ptr<SAMRAI::hier::Variable>
register_variable_Cell(mesh::Attribute *item,
                       int ndims,
                       SAMRAI::hier::IntVector const &d_nghosts,
                       SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                       boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &d_grid_geometry,
                       boost::shared_ptr<SAMRAI::appu::VisItDataWriter> const &d_visit_writer,
                       SAMRAI::hier::VariableDatabase *vardb)
{
    SAMRAI::tbox::Dimension d_dim(ndims);

    auto var = boost::make_shared<SAMRAI::pdat::CellVariable<T>>(d_dim, item->name(), item->value_size());
    integrator->registerVariable(var, d_nghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                 d_grid_geometry,
                                 "CONSERVATIVE_COARSEN",
                                 "CONSERVATIVE_LINEAR_REFINE");
    d_visit_writer->registerPlotQuantity(item->name(), visit_variable_type[item->value_rank()],
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

void SAMRAIWorker::registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator)
{

    ASSERT(integrator != nullptr);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

    integrator->registerVariable(d_uval, d_nghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                 d_grid_geometry,
                                 "CONSERVATIVE_COARSEN",
                                 "CONSERVATIVE_LINEAR_REFINE");

    integrator->registerVariable(d_flux, d_fluxghosts,
                                 SAMRAI::algs::HyperbolicLevelIntegrator::FLUX,
                                 d_grid_geometry,
                                 "CONSERVATIVE_COARSEN",
                                 "NO_REFINE");
    d_visit_writer->registerPlotQuantity("U",
                                         "SCALAR",
                                         vardb->mapVariableAndContextToIndex(
                                                 d_uval, integrator->getPlotContext()));

    if (!d_visit_writer)
    {
        RUNTIME_ERROR << m_name_ << ": registerModelVariables() VisIt data writer was not registered."
                "Consequently, no plot data will be written." << std::endl;
    }

    m_worker_->for_each(
            [&](mesh::Worker::Observer &ob)
            {
                mesh::Attribute *item = ob.attribute();
                if (item == nullptr) { return; }
                auto var = detail::register_variable(item, d_dim.getValue(),
                                                     (item->entity_type() == mesh::VERTEX ||
                                                      item->entity_type() == mesh::VOLUME) ? d_nghosts : d_fluxghosts,
                                                     integrator,
                                                     d_grid_geometry,
                                                     d_visit_writer,
                                                     vardb
                );
                m_samrai_variables_.emplace(std::pair(item->id, var));
            }
    );
//    vardb->printClassData(std::cout);

}


/*
 *************************************************************************
 *
 * Set up parameters for nonuniform load balancing, if used.
 *
 *************************************************************************
 */

void SAMRAIWorker::setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                                     SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm)
{

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
            WARNING << m_name_ << ": "
                    << "  Unknown load balancer used in gridding algorithm."
                    << "  Ignoring request for nonuniform load balancing." << std::endl;
            d_use_nonuniform_workload = false;
        }
    } else
    {
        d_use_nonuniform_workload = false;
    }

}

namespace detail
{
mesh::DataBlock *
convert_patch_data(mesh::Attribute *attr, boost::shared_ptr<SAMRAI::hier::PatchData> const &p_data)
{
}

mesh::MeshBlock *
convert_patch_mesh(mesh::Atlas *attr, SAMRAI::hier::Patch &p)
{

}


}//namespace detail


void
SAMRAIWorker::move_to(mesh::Worker *w, SAMRAI::hier::Patch &patch)
{
    mesh::MeshBlock const *m = detail::convert_patch_mesh(m_atlas_.get(), patch);

    m_worker_->for_each(
            [&](mesh::Worker::Observer &ob)
            {
                mesh::Attribute *attr = ob.attribute();
                if (attr == nullptr) { return; }

                ob.move_to(m,
                           detail::convert_patch_data(attr,
                                                      patch.getPatchData(m_samrai_variables_.at(attr->id()),
                                                                         getDataContext())));
            }
    );
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
void SAMRAIWorker::initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time,
                                         const bool initial_time)
{
    move_to(m_worker_.get(), patch);

    m_worker_->initialize(data_time, initial_time);

    if (d_use_nonuniform_workload)
    {
        if (!patch.checkAllocated(d_workload_data_id))
        {
            patch.allocatePatchData(d_workload_data_id);
        }
        boost::shared_ptr<SAMRAI::pdat::CellData<double>> workload_data(
                boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
                        patch.getPatchData(d_workload_data_id)));
        TBOX_ASSERT(workload_data);

        const SAMRAI::hier::Box &box = patch.getBox();
        const SAMRAI::hier::BoxId &box_id = box.getBoxId();
        const SAMRAI::hier::LocalId &local_id = box_id.getLocalId();
        double id_val = local_id.getValue() % 2 ? static_cast<double>(local_id.getValue() % 10) : 0.0;
        workload_data->fillAll(1.0 + id_val);
    }

}

/*
 *************************************************************************
 *
 * Compute stable time increment for patch.  Return this m_value_.
 *
 *************************************************************************
 */

double SAMRAIWorker::computeStableDtOnPatch(
        SAMRAI::hier::Patch &patch,
        const bool initial_time,
        const double dt_time)
{
    move_to(m_worker_.get(), patch);

    return m_worker_->computeStableDtOnPatch(dt_time, initial_time);

}

/*
 *************************************************************************
 *
 * Compute time integral of numerical fluxes for finite difference
 * at each cell face on patch.  When d_dim == tbox::Dimension(3)), there are two options
 * for the transverse flux correction.  Otherwise, there is only one.
 *
 *************************************************************************
 */

void SAMRAIWorker::computeFluxesOnPatch(SAMRAI::hier::Patch &patch, const double time, const double dt)
{
    move_to(m_worker_.get(), patch);

    return m_worker_->computeFluxesOnPatch(time, dt);

    return;

}


/*
 *************************************************************************
 *
 * Update solution variables by performing a conservative
 * difference with the fluxes calculated in computeFluxesOnPatch().
 *
 *************************************************************************
 */

void SAMRAIWorker::conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, const double time,
                                                 const double dt, bool at_syncronization)
{
    move_to(m_worker_.get(), patch);
    m_worker_->conservativeDifferenceOnPatch(time, dt, at_syncronization);
}

/*
 *************************************************************************
 *
 * Reset physical boundary values for special cases, such as those
 * involving symmetric (i.e., reflective) boundary conditions and
 * when the "STEP" problem is run.
 *
 *************************************************************************
 */
void SAMRAIWorker::boundaryReset(
        SAMRAI::hier::Patch &patch,
        SAMRAI::pdat::FaceData<double> &traced_left,
        SAMRAI::pdat::FaceData<double> &traced_right) const
{
    const SAMRAI::hier::Index ifirst = patch.getBox().lower();
    const SAMRAI::hier::Index ilast = patch.getBox().upper();
    int idir;
    bool bdry_cell = true;

    const boost::shared_ptr<SAMRAI::geom::CartesianPatchGeometry> patch_geom(
            boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry, SAMRAI::hier::PatchGeometry>(
                    patch.getPatchGeometry()));
    TBOX_ASSERT(patch_geom);
    SAMRAI::hier::BoxContainer domain_boxes;
    d_grid_geometry->computePhysicalDomain(domain_boxes,
                                           patch_geom->getRatio(),
                                           SAMRAI::hier::BlockId::zero());

    SAMRAI::pdat::CellIndex icell(ifirst);
    SAMRAI::hier::BoxContainer bdrybox;
    SAMRAI::hier::Index ibfirst = ifirst;
    SAMRAI::hier::Index iblast = ilast;
    int bdry_case = 0;
    int bside;

    for (idir = 0; idir < d_dim.getValue(); ++idir)
    {
        ibfirst(idir) = ifirst(idir) - 1;
        iblast(idir) = ifirst(idir) - 1;
        bdrybox.pushBack(SAMRAI::hier::Box(ibfirst, iblast, SAMRAI::hier::BlockId(0)));

        ibfirst(idir) = ilast(idir) + 1;
        iblast(idir) = ilast(idir) + 1;
        bdrybox.pushBack(SAMRAI::hier::Box(ibfirst, iblast, SAMRAI::hier::BlockId(0)));
    }

    SAMRAI::hier::BoxContainer::iterator ib = bdrybox.begin();
    for (idir = 0; idir < d_dim.getValue(); ++idir)
    {
        bside = 2 * idir;
        if (d_dim == SAMRAI::tbox::Dimension(2))
        {
            bdry_case = d_scalar_bdry_edge_conds[bside];
        }
        if (d_dim == SAMRAI::tbox::Dimension(3))
        {

            bdry_case = d_scalar_bdry_face_conds[bside];
        }
        if (bdry_case == BdryCond::REFLECT)
        {
            SAMRAI::pdat::CellIterator icend(SAMRAI::pdat::CellGeometry::end(*ib));
            for (SAMRAI::pdat::CellIterator ic(SAMRAI::pdat::CellGeometry::begin(*ib));
                 ic != icend; ++ic)
            {
                for (SAMRAI::hier::BoxContainer::iterator domain_boxes_itr =
                        domain_boxes.begin();
                     domain_boxes_itr != domain_boxes.end();
                     ++domain_boxes_itr)
                {
                    if (domain_boxes_itr->contains(*ic))
                        bdry_cell = false;
                }
                if (bdry_cell)
                {
                    SAMRAI::pdat::FaceIndex sidein = SAMRAI::pdat::FaceIndex(*ic, idir, 1);
                    (traced_left)(sidein, 0) = (traced_right)(sidein, 0);
                }
            }
        }
        ++ib;

        int bnode = 2 * idir + 1;
        if (d_dim == SAMRAI::tbox::Dimension(2))
        {
            bdry_case = d_scalar_bdry_edge_conds[bnode];
        }
        if (d_dim == SAMRAI::tbox::Dimension(3))
        {
            bdry_case = d_scalar_bdry_face_conds[bnode];
        }
        if (bdry_case == BdryCond::REFLECT)
        {
            SAMRAI::pdat::CellIterator icend(SAMRAI::pdat::CellGeometry::end(*ib));
            for (SAMRAI::pdat::CellIterator ic(SAMRAI::pdat::CellGeometry::begin(*ib));
                 ic != icend; ++ic)
            {
                for (SAMRAI::hier::BoxContainer::iterator domain_boxes_itr =
                        domain_boxes.begin();
                     domain_boxes_itr != domain_boxes.end();
                     ++domain_boxes_itr)
                {
                    if (domain_boxes_itr->contains(*ic))
                        bdry_cell = false;
                }
                if (bdry_cell)
                {
                    SAMRAI::pdat::FaceIndex sidein = SAMRAI::pdat::FaceIndex(*ic, idir, 0);
                    (traced_right)(sidein, 0) = (traced_left)(sidein, 0);
                }
            }
        }
        ++ib;
    }
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

void SAMRAIWorker::setPhysicalBoundaryConditions(
        SAMRAI::hier::Patch &patch,
        const double fill_time,
        const SAMRAI::hier::IntVector &ghost_width_to_fill)
{
}

/*
 *************************************************************************
 *
 * Tag cells for refinement using Richardson extrapolation.  Criteria
 * defined in input.
 *
 *************************************************************************
 */
void SAMRAIWorker::tagRichardsonExtrapolationCells(
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
//    INFORM << "tagRichardsonExtrapolationCells" << patch.getPatchLevelNumber() << std::endl;

//    NULL_USE(initial_error);
//
//    SAMRAI::hier::Box pbox = patch.getBox();
//
//    boost::shared_ptr<SAMRAI::pdat::CellData<int> > tags(
//            boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int>, SAMRAI::hier::PatchData>(
//                    patch.getPatchData(tag_index)));
//    TBOX_ASSERT(tags);
//
//    /*
//     * Possible tagging criteria includes
//     *    UVAL_RICHARDSON
//     * The criteria is specified over a time interval.
//     *
//     * Loop over criteria provided and check to make sure we are in the
//     * specified time interval.  If so, apply appropriate tagging for
//     * the level.
//     */
//    for (int ncrit = 0;
//         ncrit < static_cast<int>(d_refinement_criteria.size()); ++ncrit)
//    {
//
//        std::string ref = d_refinement_criteria[ncrit];
//        int size;
//        double tol;
//        bool time_allowed;
//
//        if (ref == "UVAL_RICHARDSON")
//        {
//            boost::shared_ptr<SAMRAI::pdat::CellData<double> > coarsened_fine_var =
//                    boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
//                            patch.getPatchData(d_uval, coarsened_fine));
//            boost::shared_ptr<SAMRAI::pdat::CellData<double> > advanced_coarse_var =
//                    boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
//                            patch.getPatchData(d_uval, advanced_coarse));
//            size = static_cast<int>(d_rich_tol.size());
//            tol = ((error_level_number < size)
//                   ? d_rich_tol[error_level_number]
//                   : d_rich_tol[size - 1]);
//            size = static_cast<int>(d_rich_time_min.size());
//            double time_min = ((error_level_number < size)
//                               ? d_rich_time_min[error_level_number]
//                               : d_rich_time_min[size - 1]);
//            size = static_cast<int>(d_rich_time_max.size());
//            double time_max = ((error_level_number < size)
//                               ? d_rich_time_max[error_level_number]
//                               : d_rich_time_max[size - 1]);
//            time_allowed = (time_min <= regrid_time) && (time_max > regrid_time);
//
//            if (time_allowed)
//            {
//
//                TBOX_ASSERT(coarsened_fine_var);
//                TBOX_ASSERT(advanced_coarse_var);
//                /*
//                 * We tag wherever the global error > specified tolerance
//                 * (i.e. d_rich_tol).  The estimated global error is the
//                 * local truncation error * the approximate number of steps
//                 * used in the simulation.  Approximate the number of steps as:
//                 *
//                 *       steps = L / (s*deltat)
//                 * where
//                 *       L = length of problem domain
//                 *       s = wave speed
//                 *       delta t = timestep on current level
//                 *
//                 */
//                const double *xdomainlo = d_grid_geometry->getXLower();
//                const double *xdomainhi = d_grid_geometry->getXUpper();
//                double max_length = 0.;
//                double max_wave_speed = 0.;
//                for (int idir = 0; idir < d_dim.getValue(); ++idir)
//                {
//                    double length = xdomainhi[idir] - xdomainlo[idir];
//                    if (length > max_length) max_length = length;
//
//                    double wave_speed = d_advection_velocity[idir];
//                    if (wave_speed > max_wave_speed) max_wave_speed = wave_speed;
//                }
//
//                double steps = max_length / (max_wave_speed * deltat);
//
//                /*
//                 * Tag cells where |w_c - w_f| * (r^n -1) * steps
//                 *
//                 * where
//                 *       w_c = soln on coarse level (pressure_crse)
//                 *       w_f = soln on fine level (pressure_fine)
//                 *       r   = error coarsen ratio
//                 *       n   = spatial order of scheme (1st or 2nd depending
//                 *             on whether Godunov order is 1st or 2nd/4th)
//                 */
//                int order = 1;
//                if (d_godunov_order > 1) order = 2;
//                double r = error_coarsen_ratio;
//                double rnminus1 = std::pow(r, order) - 1;
//
//                double diff = 0.;
//                double error = 0.;
//
//                SAMRAI::pdat::CellIterator icend(SAMRAI::pdat::CellGeometry::end(pbox));
//                for (SAMRAI::pdat::CellIterator ic(SAMRAI::pdat::CellGeometry::begin(pbox));
//                     ic != icend; ++ic)
//                {
//
//                    /*
//                     * Compute error norm
//                     */
//                    diff = (*advanced_coarse_var)(*ic, 0)
//                           - (*coarsened_fine_var)(*ic, 0);
//                    error = SAMRAI::tbox::MathUtilities<double>::Abs(diff) * rnminus1 * steps;
//
//                    /*
//                     * Tag cell if error > prescribed threshold. Since we are
//                     * operating on the actual tag values (not temporary ones)
//                     * distinguish here tags that were previously set before
//                     * coming into this routine and those that are set here.
//                     *     RICHARDSON_ALREADY_TAGGED - tagged before coming
//                     *                                 into this method.
//                     *     RICHARDSON_NEWLY_TAGGED - newly tagged in this method
//                     *
//                     */
//                    if (error > tol)
//                    {
//                        if ((*tags)(*ic, 0))
//                        {
//                            (*tags)(*ic, 0) = RICHARDSON_ALREADY_TAGGED;
//                        } else
//                        {
//                            (*tags)(*ic, 0) = RICHARDSON_NEWLY_TAGGED;
//                        }
//                    }
//
//                }
//
//            } // time_allowed
//
//        } // if UVAL_RICHARDSON
//
//    } // loop over refinement criteria
//
//    /*
//     * If we are NOT performing gradient detector (i.e. only
//     * doing Richardson extrapolation) set tags marked in this method
//     * to TRUE and all others false.  Otherwise, leave tags set to the
//     * RICHARDSON_ALREADY_TAGGED and RICHARDSON_NEWLY_TAGGED as we may
//     * use this information in the gradient detector.
//     */
//    if (!uses_gradient_detector_too)
//    {
//        SAMRAI::pdat::CellIterator icend(SAMRAI::pdat::CellGeometry::end(pbox));
//        for (SAMRAI::pdat::CellIterator ic(SAMRAI::pdat::CellGeometry::begin(pbox));
//             ic != icend; ++ic)
//        {
//            if ((*tags)(*ic, 0) == RICHARDSON_ALREADY_TAGGED ||
//                (*tags)(*ic, 0) == RICHARDSON_NEWLY_TAGGED)
//            {
//                (*tags)(*ic, 0) = TRUE;
//            } else
//            {
//                (*tags)(*ic, 0) = FALSE;
//            }
//        }
//    }

}

/*
 *************************************************************************
 *
 * Tag cells for refinement using gradient detector.  Tagging criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAIWorker::tagGradientDetectorCells(
        SAMRAI::hier::Patch &patch,
        const double regrid_time,
        const bool initial_error,
        const int tag_indx,
        const bool uses_richardson_extrapolation_too)
{
//    INFORM << "tagGradientDetectorCells" << patch.getPatchLevelNumber() << std::endl;

//    NULL_USE(initial_error);
//
//    const int error_level_number = patch.getPatchLevelNumber();
//
//    const boost::shared_ptr<geom::CartesianPatchGeometry> patch_geom(
//             boost::dynamic_pointer_cast<geom::CartesianPatchGeometry, hier::PatchGeometry>(
//                    patch.getPatchGeometry()));
//    TBOX_ASSERT(patch_geom);
//    const double *dx = patch_geom->getDx();
//
//    boost::shared_ptr<pdat::CellData<int> > tags(
//             boost::dynamic_pointer_cast<pdat::CellData<int>, hier::PatchData>(
//                    patch.getPatchData(tag_indx)));
//    TBOX_ASSERT(tags);
//
//    hier::Box pbox(patch.getBox());
//    hier::BoxContainer domain_boxes;
//    d_grid_geometry->computePhysicalDomain(domain_boxes,
//                                           patch_geom->getRatio(),
//                                           hier::BlockId::zero());
//    /*
//     * Construct domain bounding box
//     */
//    hier::Box domain(d_dim);
//    for (hier::BoxContainer::iterator i = domain_boxes.begin();
//         i != domain_boxes.end(); ++i)
//    {
//        domain += *i;
//    }
//
//    const hier::Index domfirst(domain.lower());
//    const hier::Index domlast(domain.upper());
//    const hier::Index ifirst(patch.getBox().lower());
//    const hier::Index ilast(patch.getBox().upper());
//
//    hier::Index ict(d_dim);
//
//    int not_refine_tag_val = FALSE;
//    int refine_tag_val = TRUE;
//
//    /*
//     * Create a set of temporary tags and set to untagged m_value_.
//     */
//    boost::shared_ptr<pdat::CellData<int> > temp_tags(
//            new pdat::CellData<int>(pbox, 1, d_nghosts));
//    temp_tags->fillAll(not_refine_tag_val);
//
//    /*
//     * Possible tagging criteria includes
//     *    UVAL_DEVIATION, UVAL_GRADIENT, UVAL_SHOCK
//     * The criteria is specified over a time interval.
//     *
//     * Loop over criteria provided and check to make sure we are in the
//     * specified time interval.  If so, apply appropriate tagging for
//     * the level.
//     */
//    for (int ncrit = 0;
//         ncrit < static_cast<int>(d_refinement_criteria.size()); ++ncrit)
//    {
//
//        string ref = d_refinement_criteria[ncrit];
//        boost::shared_ptr<pdat::CellData<double> > var(
//                 boost::dynamic_pointer_cast<pdat::CellData<double>, hier::PatchData>(
//                        patch.getPatchData(d_uval, getDataContext())));
//        TBOX_ASSERT(var);
//
//        hier::IntVector vghost(var->getGhostCellWidth());
//        hier::IntVector tagghost(tags->getGhostCellWidth());
//
//        int size = 0;
//        double tol = 0.;
//        double onset = 0.;
//        bool time_allowed = false;
//
//        if (ref == "UVAL_DEVIATION")
//        {
//            size = static_cast<int>(d_dev_tol.size());
//            tol = ((error_level_number < size)
//                   ? d_dev_tol[error_level_number]
//                   : d_dev_tol[size - 1]);
//            size = static_cast<int>(d_dev.size());
//            double dev = ((error_level_number < size)
//                          ? d_dev[error_level_number]
//                          : d_dev[size - 1]);
//            size = static_cast<int>(d_dev_time_min.size());
//            double time_min = ((error_level_number < size)
//                               ? d_dev_time_min[error_level_number]
//                               : d_dev_time_min[size - 1]);
//            size = static_cast<int>(d_dev_time_max.size());
//            double time_max = ((error_level_number < size)
//                               ? d_dev_time_max[error_level_number]
//                               : d_dev_time_max[size - 1]);
//            time_allowed = (time_min <= regrid_time) && (time_max > regrid_time);
//
//            if (time_allowed)
//            {
//
//                /*
//                 * Check for tags that have already been set in a previous
//                 * step.  Do NOT consider values tagged with m_value_
//                 * RICHARDSON_NEWLY_TAGGED since these were set most recently
//                 * by Richardson extrapolation.
//                 */
//                pdat::CellIterator icend(pdat::CellGeometry::end(pbox));
//                for (pdat::CellIterator ic(pdat::CellGeometry::begin(pbox));
//                     ic != icend; ++ic)
//                {
//                    double locden = tol;
//                    int tag_val = (*tags)(*ic, 0);
//                    if (tag_val)
//                    {
//                        if (tag_val != RICHARDSON_NEWLY_TAGGED)
//                        {
//                            locden *= 0.75;
//                        }
//                    }
//                    if (tbox::MathUtilities<double>::Abs((*var)(*ic) - dev) >
//                        locden)
//                    {
//                        (*temp_tags)(*ic, 0) = refine_tag_val;
//                    }
//                }
//            }
//        }
//
//        if (ref == "UVAL_GRADIENT")
//        {
//            size = static_cast<int>(d_grad_tol.size());
//            tol = ((error_level_number < size)
//                   ? d_grad_tol[error_level_number]
//                   : d_grad_tol[size - 1]);
//            size = static_cast<int>(d_grad_time_min.size());
//            double time_min = ((error_level_number < size)
//                               ? d_grad_time_min[error_level_number]
//                               : d_grad_time_min[size - 1]);
//            size = static_cast<int>(d_grad_time_max.size());
//            double time_max = ((error_level_number < size)
//                               ? d_grad_time_max[error_level_number]
//                               : d_grad_time_max[size - 1]);
//            time_allowed = (time_min <= regrid_time) && (time_max > regrid_time);

//            if (time_allowed)
//            {
//
//                if (d_dim == tbox::Dimension(2))
//                {
//                    SAMRAI_F77_FUNC(detectgrad2d, DETECTGRAD2D)(
//                            ifirst(0), ilast(0), ifirst(1), ilast(1),
//                            vghost(0), tagghost(0), d_nghosts(0),
//                            vghost(1), tagghost(1), d_nghosts(1),
//                            dx,
//                            tol,
//                            refine_tag_val, not_refine_tag_val,
//                            var->getPointer(),
//                            tags->getPointer(), temp_tags->getPointer());
//                }
//                if (d_dim == tbox::Dimension(3))
//                {
//                    SAMRAI_F77_FUNC(detectgrad3d, DETECTGRAD3D)(
//                            ifirst(0), ilast(0), ifirst(1), ilast(1), ifirst(2), ilast(2),
//                            vghost(0), tagghost(0), d_nghosts(0),
//                            vghost(1), tagghost(1), d_nghosts(1),
//                            vghost(2), tagghost(2), d_nghosts(2),
//                            dx,
//                            tol,
//                            refine_tag_val, not_refine_tag_val,
//                            var->getPointer(),
//                            tags->getPointer(), temp_tags->getPointer());
//                }
//            }
//
//        }
//
//        if (ref == "UVAL_SHOCK")
//        {
//            size = static_cast<int>(d_shock_tol.size());
//            tol = ((error_level_number < size)
//                   ? d_shock_tol[error_level_number]
//                   : d_shock_tol[size - 1]);
//            size = static_cast<int>(d_shock_onset.size());
//            onset = ((error_level_number < size)
//                     ? d_shock_onset[error_level_number]
//                     : d_shock_onset[size - 1]);
//            size = static_cast<int>(d_shock_time_min.size());
//            double time_min = ((error_level_number < size)
//                               ? d_shock_time_min[error_level_number]
//                               : d_shock_time_min[size - 1]);
//            size = static_cast<int>(d_shock_time_max.size());
//            double time_max = ((error_level_number < size)
//                               ? d_shock_time_max[error_level_number]
//                               : d_shock_time_max[size - 1]);
//            time_allowed = (time_min <= regrid_time) && (time_max > regrid_time);
//
//            if (time_allowed)
//            {
//
//                if (d_dim == tbox::Dimension(2))
//                {
//                    SAMRAI_F77_FUNC(detectshock2d, DETECTSHOCK2D)(
//                            ifirst(0), ilast(0), ifirst(1), ilast(1),
//                            vghost(0), tagghost(0), d_nghosts(0),
//                            vghost(1), tagghost(1), d_nghosts(1),
//                            dx,
//                            tol,
//                            onset,
//                            refine_tag_val, not_refine_tag_val,
//                            var->getPointer(),
//                            tags->getPointer(), temp_tags->getPointer());
//                }
//                if (d_dim == tbox::Dimension(3))
//                {
//                    SAMRAI_F77_FUNC(detectshock3d, DETECTSHOCK3D)(
//                            ifirst(0), ilast(0), ifirst(1), ilast(1), ifirst(2), ilast(2),
//                            vghost(0), tagghost(0), d_nghosts(0),
//                            vghost(1), tagghost(1), d_nghosts(1),
//                            vghost(2), tagghost(2), d_nghosts(2),
//                            dx,
//                            tol,
//                            onset,
//                            refine_tag_val, not_refine_tag_val,
//                            var->getPointer(),
//                            tags->getPointer(), temp_tags->getPointer());
//                }
//            }
//
//        }
//
//    }  // loop over criteria
//
//    /*
//     * Adjust temp_tags from those tags set in Richardson extrapolation.
//     * Here, we just reset any tags that were set in Richardson extrapolation
//     * to be the designated "refine_tag_val".
//     */
//    if (uses_richardson_extrapolation_too)
//    {
//        pdat::CellIterator icend(pdat::CellGeometry::end(pbox));
//        for (pdat::CellIterator ic(pdat::CellGeometry::begin(pbox));
//             ic != icend; ++ic)
//        {
//            if ((*tags)(*ic, 0) == RICHARDSON_ALREADY_TAGGED ||
//                (*tags)(*ic, 0) == RICHARDSON_NEWLY_TAGGED)
//            {
//                (*temp_tags)(*ic, 0) = refine_tag_val;
//            }
//        }
//    }
//
//    /*
//     * Update tags.
//     */
//    pdat::CellIterator icend(pdat::CellGeometry::end(pbox));
//    for (pdat::CellIterator ic(pdat::CellGeometry::begin(pbox));
//         ic != icend; ++ic)
//    {
//        (*tags)(*ic, 0) = (*temp_tags)(*ic, 0);
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

#ifdef HAVE_HDF5

void SAMRAIWorker::registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer)
{
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}

#endif

/*
 *************************************************************************
 *
 * Write SAMRAIWorker object state to specified stream.
 *
 *************************************************************************
 */

void SAMRAIWorker::printClassData(std::ostream &os) const
{
    int j, k;

    os << "\nSAMRAIWorker::printClassData..." << std::endl;
    os << "SAMRAIWorker: this = " << (SAMRAIWorker *)
            this << std::endl;
    os << "m_name_ = " << m_name_ << std::endl;
    os << "d_grid_geometry = "
       << d_grid_geometry.get() << std::endl;

    os << "Parameters for numerical method ..." << std::endl;
    os << "   d_advection_velocity = ";
    for (j = 0; j < d_dim.getValue(); ++j) os << d_advection_velocity[j] << " ";
    os << std::endl;
    os << "   d_godunov_order = " << d_godunov_order << std::endl;
    os << "   d_corner_transport = " << d_corner_transport << std::endl;
    os << "   d_nghosts = " << d_nghosts << std::endl;
    os << "   d_fluxghosts = " << d_fluxghosts << std::endl;

    os << "Problem description and initial data..." << std::endl;
    os << "   d_data_problem = " << d_data_problem << std::endl;
    os << "   d_data_problem_int = " << d_data_problem << std::endl;

    os << "       d_radius = " << d_radius << std::endl;
    os << "       d_center = ";
    for (j = 0; j < d_dim.getValue(); ++j) os << d_center[j] << " ";
    os << std::endl;
    os << "       d_uval_inside = " << d_uval_inside << std::endl;
    os << "       d_uval_outside = " << d_uval_outside << std::endl;

    os << "       d_number_of_intervals = " << d_number_of_intervals << std::endl;
    os << "       d_front_position = ";
    for (k = 0; k < d_number_of_intervals - 1; ++k)
    {
        os << d_front_position[k] << "  ";
    }
    os << std::endl;
    os << "       d_interval_uval = " << std::endl;
    for (k = 0; k < d_number_of_intervals; ++k)
    {
        os << "            " << d_interval_uval[k] << std::endl;
    }
    os << "   Boundary condition data " << std::endl;

    if (d_dim == SAMRAI::tbox::Dimension(2))
    {
        for (j = 0; j < static_cast<int>(d_scalar_bdry_edge_conds.size()); ++j)
        {
            os << "       d_scalar_bdry_edge_conds[" << j << "] = "
               << d_scalar_bdry_edge_conds[j] << std::endl;
            if (d_scalar_bdry_edge_conds[j] == BdryCond::DIRICHLET)
            {
                os << "         d_bdry_edge_uval[" << j << "] = "
                   << d_bdry_edge_uval[j] << std::endl;
            }
        }
        os << std::endl;
        for (j = 0; j < static_cast<int>(d_scalar_bdry_node_conds.size()); ++j)
        {
            os << "       d_scalar_bdry_node_conds[" << j << "] = "
               << d_scalar_bdry_node_conds[j] << std::endl;
            os << "       d_node_bdry_edge[" << j << "] = "
               << d_node_bdry_edge[j] << std::endl;
        }
    }
    if (d_dim == SAMRAI::tbox::Dimension(3))
    {
        for (j = 0; j < static_cast<int>(d_scalar_bdry_face_conds.size()); ++j)
        {
            os << "       d_scalar_bdry_face_conds[" << j << "] = "
               << d_scalar_bdry_face_conds[j] << std::endl;
            if (d_scalar_bdry_face_conds[j] == BdryCond::DIRICHLET)
            {
                os << "         d_bdry_face_uval[" << j << "] = "
                   << d_bdry_face_uval[j] << std::endl;
            }
        }
        os << std::endl;
        for (j = 0; j < static_cast<int>(d_scalar_bdry_edge_conds.size()); ++j)
        {
            os << "       d_scalar_bdry_edge_conds[" << j << "] = "
               << d_scalar_bdry_edge_conds[j] << std::endl;
            os << "       d_edge_bdry_face[" << j << "] = "
               << d_edge_bdry_face[j] << std::endl;
        }
        os << std::endl;
        for (j = 0; j < static_cast<int>(d_scalar_bdry_node_conds.size()); ++j)
        {
            os << "       d_scalar_bdry_node_conds[" << j << "] = "
               << d_scalar_bdry_node_conds[j] << std::endl;
            os << "       d_node_bdry_face[" << j << "] = "
               << d_node_bdry_face[j] << std::endl;
        }
    }

    os << "   Refinement criteria parameters " << std::endl;

    for (j = 0; j < static_cast<int>(d_refinement_criteria.size()); ++j)
    {
        os << "       d_refinement_criteria[" << j << "] = "
           << d_refinement_criteria[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_dev_tol.size()); ++j)
    {
        os << "       d_dev_tol[" << j << "] = "
           << d_dev_tol[j] << std::endl;
    }
    for (j = 0; j < static_cast<int>(d_dev.size()); ++j)
    {
        os << "       d_dev[" << j << "] = "
           << d_dev[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_dev_time_max.size()); ++j)
    {
        os << "       d_dev_time_max[" << j << "] = "
           << d_dev_time_max[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_dev_time_min.size()); ++j)
    {
        os << "       d_dev_time_min[" << j << "] = "
           << d_dev_time_min[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_grad_tol.size()); ++j)
    {
        os << "       d_grad_tol[" << j << "] = "
           << d_grad_tol[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_grad_time_max.size()); ++j)
    {
        os << "       d_grad_time_max[" << j << "] = "
           << d_grad_time_max[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_grad_time_min.size()); ++j)
    {
        os << "       d_grad_time_min[" << j << "] = "
           << d_grad_time_min[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_shock_onset.size()); ++j)
    {
        os << "       d_shock_onset[" << j << "] = "
           << d_shock_onset[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_shock_tol.size()); ++j)
    {
        os << "       d_shock_tol[" << j << "] = "
           << d_shock_tol[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_shock_time_max.size()); ++j)
    {
        os << "       d_shock_time_max[" << j << "] = "
           << d_shock_time_max[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_shock_time_min.size()); ++j)
    {
        os << "       d_shock_time_min[" << j << "] = "
           << d_shock_time_min[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_rich_tol.size()); ++j)
    {
        os << "       d_rich_tol[" << j << "] = "
           << d_rich_tol[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_rich_time_max.size()); ++j)
    {
        os << "       d_rich_time_max[" << j << "] = "
           << d_rich_time_max[j] << std::endl;
    }
    os << std::endl;
    for (j = 0; j < static_cast<int>(d_rich_time_min.size()); ++j)
    {
        os << "       d_rich_time_min[" << j << "] = "
           << d_rich_time_min[j] << std::endl;
    }
    os << std::endl;

}

/*
 *************************************************************************
 *
 * Routines to read boundary data from input database.
 *
 *************************************************************************
 */

void SAMRAIWorker::readDirichletBoundaryDataEntry(const boost::shared_ptr<SAMRAI::tbox::Database> &db,
                                                  std::string &db_name,
                                                  int bdry_location_index)
{
//    TBOX_ASSERT(db);
//    TBOX_ASSERT(!db_name.empty());
//
//    if (d_dim == SAMRAI::tbox::Dimension(2))
//    {
//        readStateDataEntry(db,
//                           db_name,
//                           bdry_location_index,
//                           d_bdry_edge_uval);
//    }
//    if (d_dim == SAMRAI::tbox::Dimension(3))
//    {
//        readStateDataEntry(db,
//                           db_name,
//                           bdry_location_index,
//                           d_bdry_face_uval);
//    }
}

void SAMRAIWorker::readNeumannBoundaryDataEntry(const boost::shared_ptr<SAMRAI::tbox::Database> &db,
                                                std::string &db_name,
                                                int bdry_location_index)
{
    NULL_USE(db);
    NULL_USE(db_name);
    NULL_USE(bdry_location_index);
}

struct SAMRAILevelIntegrator : public SAMRAI::algs::HyperbolicLevelIntegrator
{
    template<typename ...Args>
    SAMRAILevelIntegrator(Args &&...args):SAMRAI::algs::HyperbolicLevelIntegrator(std::forward<Args>(args)...) {}

    virtual ~SAMRAILevelIntegrator() {}
};

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

    boost::shared_ptr<SAMRAI::tbox::Database> samrai_cfg;

    boost::shared_ptr<SAMRAIWorker> patch_worker;

    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;

    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;

    boost::shared_ptr<SAMRAILevelIntegrator> hyp_level_integrator;

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
    hyp_level_integrator->printClassData(os);
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

    samrai_cfg = detail::convert_database(db, name());



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
    /***
     *  create hyp_level_integrator and error_detector
     */
    patch_worker = boost::make_shared<SAMRAIWorker>(m_worker_, dim, grid_geometry);

    hyp_level_integrator = boost::make_shared<SAMRAILevelIntegrator>(
            "SAMRAILevelIntegrator", samrai_cfg->getDatabase("HyperbolicLevelIntegrator"),
            patch_worker.get(), use_refined_timestepping);

//    hyp_level_integrator->printClassData(std::cout);

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
            "StandardTagAndInitialize", hyp_level_integrator.get(),
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
            hyp_level_integrator,
            gridding_algorithm);


    visit_data_writer = boost::make_shared<SAMRAI::appu::VisItDataWriter>(
            dim,
            db["output_writer_name"].as<std::string>(name() + " VisIt Writer"),
            db["output_dir_name"].as<std::string>(name()),
            db["visit_number_procs_per_file"].as<int>(1)
    );

    patch_worker->registerVisItDataWriter(visit_data_writer);


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

    hyp_level_integrator.reset();

    patch_worker.reset();
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
//void SAMRAIWorker::checkBoundaryData(
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
//SAMRAIWorker::checkUserTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const
//{
//    boost::shared_ptr<SAMRAI::pdat::CellData<int> > tags( boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<int>, SAMRAI::hier::PatchData>(patch.getPatchData(tag_index)));
//    TBOX_ASSERT(tags);
//}
//
//void
//SAMRAIWorker::checkNewPatchTagData(SAMRAI::hier::DataBlockBase &patch, const int tag_index) const
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