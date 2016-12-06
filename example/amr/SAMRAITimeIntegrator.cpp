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
#include <simpla/toolbox/nTupleExt.h>

#include <simpla/data/DataEntityTable.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/DataBlock.h>
#include <simpla/manifold/Worker.h>
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
#include <SAMRAI/appu/CartesianBoundaryDefines.h>
#include <SAMRAI/pdat/NodeDoubleLinearTimeInterpolateOp.h>
#include <simpla/physics/Constants.h>


namespace simpla
{
struct SAMRAIWorker;

struct SAMRAITimeIntegrator;

std::shared_ptr<simulation::TimeIntegrator>
create_time_integrator(std::string const &name)
{
    auto integrator = std::dynamic_pointer_cast<simulation::TimeIntegrator>(
            std::make_shared<SAMRAITimeIntegrator>(name));

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



    integrator->db.set_value("CartesianGeometry.domain_boxes_0", index_box_type{{0,  0,  0},
                                                                                {16, 16, 16}});

    integrator->db.set_value("CartesianGeometry.periodic_dimension", nTuple<int, 3>{1, 1, 1});
    integrator->db.set_value("CartesianGeometry.x_lo", nTuple<double, 3>{1, 0, -1});
    integrator->db.set_value("CartesianGeometry.x_up", nTuple<double, 3>{2, PI, 1});

    integrator->db.set_value("PatchHierarchy.max_levels", int(3)); // Maximum number of levels in hierarchy.
    integrator->db.set_value("PatchHierarchy.ratio_to_coarser.level_1", nTuple<int, 3>{2, 2, 1});
    integrator->db.set_value("PatchHierarchy.ratio_to_coarser.level_2", nTuple<int, 3>{2, 2, 1});
    integrator->db.set_value("PatchHierarchy.ratio_to_coarser.level_3", nTuple<int, 3>{2, 2, 1});
    integrator->db.set_value("PatchHierarchy.largest_patch_size.level_0", nTuple<int, 3>{32, 32, 32});
    integrator->db.set_value("PatchHierarchy.smallest_patch_size.level_0", nTuple<int, 3>{4, 4, 4});

    integrator->db.set_value("GriddingAlgorithm", "");


    integrator->db.set_value("BergerRigoutsos.sort_output_nodes", true);// Makes results repeatable.
    integrator->db.set_value("BergerRigoutsos.efficiency_tolerance", 0.85);  // min % of tag cells in new patch level
    integrator->db.set_value("BergerRigoutsos.combine_efficiency", 0.95);  // chop box if sum of volumes of smaller
//    // boxes < efficiency * vol of large box


    // Refer to mesh::StandardTagAndInitialize for input
    integrator->db.set_value("StandardTagAndInitialize.tagging_method", std::string("GRADIENT_DETECTOR"));

    // Refer to algs::HyperbolicLevelIntegrator for input
    integrator->db.set_value("HyperbolicLevelIntegrator.cfl", 0.9);  // max cfl factor used in problem
    integrator->db.set_value("HyperbolicLevelIntegrator.cfl_init", 0.9); // initial cfl factor
    integrator->db.set_value("HyperbolicLevelIntegrator.lag_dt_computation", true);
    integrator->db.set_value("HyperbolicLevelIntegrator.use_ghosts_to_compute_dt", true);

    // Refer to algs::TimeRefinementIntegrator for input
    integrator->db.set_value("TimeRefinementIntegrator.start_time", 0.e0); // initial simulation time
    integrator->db.set_value("TimeRefinementIntegrator.end_time", 1.e0);  // final simulation time
    integrator->db.set_value("TimeRefinementIntegrator.grow_dt", 1.1e0);  // growth factor for timesteps
    integrator->db.set_value("TimeRefinementIntegrator.max_integrator_steps", 5);  // max number of simulation timesteps

    // Refer to mesh::TreeLoadBalancer for input
    integrator->db.set_value("LoadBalancer", "");

    return integrator;


}


class SAMRAIWorker :
        public SAMRAI::algs::HyperbolicPatchStrategy
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


    /**
     * Register a VisIt data writer so this class will write
     * plot files that may be postprocessed with the VisIt
     * visualization tool.
     */
    void registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer);


    /**
     * Print all data members for SAMRAIWorkerHyperbolic class.
     */
    void printClassData(std::ostream &os) const;

private:
    void move_to(std::shared_ptr<mesh::Worker> &w, SAMRAI::hier::Patch &patch);

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
     * data_block, set_value physical boundary conditions, and register plot
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

    boost::shared_ptr<SAMRAI::pdat::NodeVariable<double> > d_xyz;

    SAMRAI::hier::IntVector d_nghosts;
    SAMRAI::hier::IntVector d_fluxghosts;


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
        d_nghosts(dim, 4),
        d_fluxghosts(dim, 1)
{
    TBOX_ASSERT(grid_geom);

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
//struct op_create {};
//template<typename TV, mesh::MeshEntityType IFORM> struct VariableTraits;
//template<typename T> struct VariableTraits<T, mesh::VERTEX> { typedef SAMRAI::pdat::NodeVariable<T> type; };
//template<typename T> struct VariableTraits<T, mesh::EDGE> { typedef SAMRAI::pdat::EdgeVariable<T> type; };
//template<typename T> struct VariableTraits<T, mesh::FACE> { typedef SAMRAI::pdat::FaceVariable<T> type; };
//template<typename T> struct VariableTraits<T, mesh::VOLUME> { typedef SAMRAI::pdat::CellVariable<T> type; };
//
//template<typename TV, mesh::MeshEntityType IFORM> void
//attr_op(mesh::Attribute *item, op_create const &, boost::shared_ptr<SAMRAI::hier::Variable> *res, int ndims)
//{
//    SAMRAI::tbox::Dimension d_dim(ndims);
//    *res = boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
//            boost::make_shared<typename VariableTraits<TV, IFORM>::type>(d_dim, item->name(), item->dof()));
//}
//
//
//template<typename T, typename ...Args> void
//attr_choice_form(mesh::Attribute *item, Args &&...args)
//{
//
//    if (item->entity_type() == mesh::VERTEX) { attr_op<T, mesh::VERTEX>(item, std::forward<Args>(args)...); }
//    else if (item->entity_type() == mesh::EDGE) { attr_op<T, mesh::EDGE>(item, std::forward<Args>(args)...); }
//    else if (item->entity_type() == mesh::FACE) { attr_op<T, mesh::FACE>(item, std::forward<Args>(args)...); }
//    else if (item->entity_type() == mesh::VOLUME) { attr_op<T, mesh::VOLUME>(item, std::forward<Args>(args)...); }
//    else { UNIMPLEMENTED; }
//
//}
//
//template<typename ...Args> void
//attr_choice(mesh::Attribute *item, Args &&...args)
//{
//    if (item->value_type_info() == typeid(float)) { attr_choice_form<float>(item, std::forward<Args>(args)...); }
//    else if (item->value_type_info() == typeid(double)) { attr_choice_form<double>(item, std::forward<Args>(args)...); }
//    else if (item->value_type_info() == typeid(int)) { attr_choice_form<int>(item, std::forward<Args>(args)...); }
////    else if (item->value_type_info() == typeid(long)) { attr_choice_form<long>(item, std::forward<Args>(args)...); }
//    else { RUNTIME_ERROR << "Unsupported m_value_ type" << std::endl; }
//
//
//};

template<typename T>
boost::shared_ptr<SAMRAI::hier::Variable>
create_samrai_variable_t(unsigned int ndims, mesh::AttributeBase *attr)
{
    static int var_depth[4] = {1, 3, 3, 1};
    if (attr->entity_type() <= mesh::VOLUME)
    {

        SAMRAI::tbox::Dimension d_dim(ndims);

        return boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
                boost::make_shared<SAMRAI::pdat::NodeVariable<T> >(
                        d_dim, attr->name(), var_depth[attr->entity_type()] * attr->dof()));
    } else
    {
        UNIMPLEMENTED;
        return nullptr;
    }
}

boost::shared_ptr<SAMRAI::hier::Variable>
create_samrai_variable(unsigned int ndims, mesh::AttributeBase *item)
{
    if (item->value_type_info() == typeid(float)) { return create_samrai_variable_t<float>(ndims, item); }
    else if (item->value_type_info() == typeid(double)) { return create_samrai_variable_t<double>(ndims, item); }
    else if (item->value_type_info() == typeid(int)) { return create_samrai_variable_t<int>(ndims, item); }
//    else if (item->value_type_info() == typeid(long)) { attr_choice_form<long>(item, std::forward<Args>(args)...); }
    else { RUNTIME_ERROR << " m_value_ type is not supported!" << std::endl; }
    return nullptr;
}
}//namespace detail{
/**
 *
 * Register conserved variables  and  register plot data with VisIt.
 *
 */
void SAMRAIWorker::registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator)
{

    ASSERT(integrator != nullptr);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

    if (!d_visit_writer)
    {
        RUNTIME_ERROR << m_name_ << ": registerModelVariables() VisIt data_block writer was not registered."
                "Consequently, no plot data_block will be written." << std::endl;
    }

    //**************************************************************
    for (auto &ob:m_worker_->get_chart()->attributes())
    {
        auto &attr = ob->attribute();
        if (attr == nullptr) { return; }

        boost::shared_ptr<SAMRAI::hier::Variable> var = detail::create_samrai_variable(3, attr.get());

        m_samrai_variables_[attr->id()] = var;

//                static const char visit_variable_type[3][10] = {"SCALAR", "VECTOR", "TENSOR"};
//                static const char visit_variable_type2[4][10] = {"SCALAR", "VECTOR", "VECTOR", "SCALAR"};


        /*** FIXME:
        *  1. SAMRAI Visit Writer only support NODE and CELL variable (double,float ,int)
        *  2. SAMRAI   SAMRAI::algs::HyperbolicLevelIntegrator->registerVariable only support double
        **/

        if (attr->db.has("config") && attr->db.get_value<std::string>("config") == "COORDINATES")
        {
            VERBOSE << attr->name() << " is registered as coordinate" << std::endl;
            integrator->registerVariable(var, d_nghosts,
                                         SAMRAI::algs::HyperbolicLevelIntegrator::INPUT,
                                         d_grid_geometry,
                                         "",
                                         "LINEAR_REFINE");

        } else if (attr->db.has("config") && attr->db.get_value<std::string>("config") == "FLUX")
        {
            integrator->registerVariable(var, d_fluxghosts,
                                         SAMRAI::algs::HyperbolicLevelIntegrator::FLUX,
                                         d_grid_geometry,
                                         "CONSERVATIVE_COARSEN",
                                         "NO_REFINE");

        } else if (attr->db.has("config") && attr->db.get_value<std::string>("config") == "INPUT")
        {
            integrator->registerVariable(var, d_nghosts,
                                         SAMRAI::algs::HyperbolicLevelIntegrator::INPUT,
                                         d_grid_geometry,
                                         "",
                                         "NO_REFINE");
        } else
        {
            switch (attr->entity_type())
            {
                case mesh::EDGE:
                case mesh::FACE:
//                            integrator->registerVariable(var, d_nghosts,
//                                                         SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
//                                                         d_grid_geometry,
//                                                         "CONSERVATIVE_COARSEN",
//                                                         "CONSERVATIVE_LINEAR_REFINE");
//                            break;
                case mesh::VERTEX:
                case mesh::VOLUME:
                default:
                    integrator->registerVariable(var, d_nghosts,
                                                 SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
                                                 d_grid_geometry,
                                                 "",
                                                 "LINEAR_REFINE");
            }

//            VERBOSE << (attr->name()) << " --  " << visit_variable_type << std::endl;


        }

        std::string visit_variable_type = "";
        if ((attr->entity_type() == mesh::VERTEX || attr->entity_type() == mesh::VOLUME) && attr->dof() == 1)
        {
            visit_variable_type = "SCALAR";
        } else if (((attr->entity_type() == mesh::EDGE || attr->entity_type() == mesh::FACE) && attr->dof() == 1) ||
                   ((attr->entity_type() == mesh::VERTEX || attr->entity_type() == mesh::VOLUME) && attr->dof() == 3))
        {
            visit_variable_type = "VECTOR";
        } else if (
                ((attr->entity_type() == mesh::VERTEX || attr->entity_type() == mesh::VOLUME) && attr->dof() == 9) ||
                ((attr->entity_type() == mesh::EDGE || attr->entity_type() == mesh::FACE) && attr->dof() == 3)
                )
        {
            visit_variable_type = "TENSOR";
        } else
        {
            WARNING << "Can not register attribute [" << attr->name() << "] to VisIt  writer!" << std::endl;
        }


        if (visit_variable_type == "" || !attr->db.has("config")) {}
        else if (attr->db.get_value<std::string>("config") == "COORDINATES")
        {
            d_visit_writer->registerNodeCoordinates(
                    vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));

        } else
        {
            d_visit_writer->registerPlotQuantity(
                    attr->name(), visit_variable_type,
                    vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
        }
    }
//    integrator->printClassData(std::cout);
//    vardb->printClassData(std::cout);
}


/**
 * Set up parameters for nonuniform load balancing, if used.
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
//struct op_convert {};
//template<typename TV, mesh::MeshEntityType IFORM> struct PatchDataTraits;
//template<typename T> struct PatchDataTraits<T, mesh::VERTEX> { typedef SAMRAI::pdat::NodeData<T> type; };
//template<typename T> struct PatchDataTraits<T, mesh::EDGE> { typedef SAMRAI::pdat::EdgeData<T> type; };
//template<typename T> struct PatchDataTraits<T, mesh::FACE> { typedef SAMRAI::pdat::FaceData<T> type; };
//template<typename T> struct PatchDataTraits<T, mesh::VOLUME> { typedef SAMRAI::pdat::CellData<T> type; };
//
//
//template<typename TV> std::shared_ptr<mesh::DataBlock>
//convert(boost::shared_ptr<SAMRAI::pdat::NodeData<TV>> p_data)
//{
//    auto lo = p_data->getGhostBox().lower();
//    auto up = p_data->getGhostBox().upper();
//    index_type i_lo[4] = {lo[0], lo[1], lo[2], 0};
//    index_type i_up[4] = {up[0] + 2, up[1] + 2, up[2] + 2, 1};
//
//    return std::dynamic_pointer_cast<mesh::DataBlock>(
//            std::make_shared<mesh::DataBlockArray<TV, mesh::VERTEX>>(p_data->getPointer(0), 3, i_lo, i_up,
//                                                                     data_block::FAST_FIRST));
//}
//
//template<typename TV> std::shared_ptr<mesh::DataBlock>
//convert(boost::shared_ptr<SAMRAI::pdat::CellData<TV>> p_data)
//{
//    auto lo = p_data->getGhostBox().lower();
//    auto up = p_data->getGhostBox().upper();
//    index_type i_lo[4] = {lo[0], lo[1], lo[2], 0};
//    index_type i_up[4] = {up[0], up[1], up[2], 1};
//
//    return std::dynamic_pointer_cast<mesh::DataBlock>(
//            std::make_shared<mesh::DataBlockArray<TV, mesh::VOLUME>>(p_data->getPointer(0), 3, i_lo, i_up,
//                                                                     data_block::FAST_FIRST));
//}
//
//template<typename TV> std::shared_ptr<mesh::DataBlock>
//convert(boost::shared_ptr<SAMRAI::pdat::EdgeData<TV>> p_data)
//{
//    auto lo = p_data->getGhostBox().lower();
//    auto up = p_data->getGhostBox().upper();
//    index_type i_lo[4] = {lo[0], lo[1], lo[2], 0};
//    index_type i_up[4] = {up[0], up[1], up[2], 3};
//
//    return std::dynamic_pointer_cast<mesh::DataBlock>(
//            std::make_shared<mesh::DataBlockArray<TV, mesh::EDGE>>(p_data->getPointer(0), 4, i_lo, i_up,
//                                                                   data_block::FAST_FIRST));
//}
//
//template<typename TV> std::shared_ptr<mesh::DataBlock>
//convert(boost::shared_ptr<SAMRAI::pdat::FaceData<TV>> p_data)
//{
//    auto lo = p_data->getGhostBox().lower();
//    auto up = p_data->getGhostBox().upper();
//
//    index_type i_lo[4] = {lo[0], lo[1], lo[2], 0};
//    index_type i_up[4] = {up[0], up[1], up[2], 3};
//
//    return std::dynamic_pointer_cast<mesh::DataBlock>(
//            std::make_shared<mesh::DataBlockArray<TV, mesh::FACE>>(p_data->getPointer(0), 4, i_lo, i_up,
//                                                                   data_block::FAST_FIRST));
//}
//
//template<typename TV, mesh::MeshEntityType IFORM> void
//attr_op(mesh::Attribute *item, op_convert const &,
//        std::shared_ptr<mesh::DataBlock> *res,
//        boost::shared_ptr<SAMRAI::hier::PatchData> p_data)
//{
//    auto pd = boost::dynamic_pointer_cast<typename PatchDataTraits<TV, IFORM>::type>(p_data);
//    pd->fillAll(0);
//    *res = convert(pd);
//}
template<typename TV, mesh::MeshEntityType IFORM, size_type DOF>
std::shared_ptr<mesh::DataBlock>
create_data_block_t2(std::shared_ptr<mesh::AttributeBase> const &item, boost::shared_ptr<SAMRAI::hier::PatchData> pd)
{
    auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::NodeData<TV>>(pd);

    int ndims = p_data->getDim().getValue();

    int depth = p_data->getDepth();


    auto outer_lower = p_data->getGhostBox().lower();
    auto outer_upper = p_data->getGhostBox().upper();

    auto inner_lower = p_data->getBox().lower();
    auto inner_upper = p_data->getBox().upper();
    index_type o_lower[4] = {outer_lower[0], outer_lower[1], outer_lower[2], 0};
    index_type o_upper[4] = {outer_upper[0] + 2, outer_upper[1] + 2, outer_upper[2] + 2, depth};

    index_type i_lower[4] = {inner_lower[0], inner_lower[1], inner_lower[2], 0};
    index_type i_upper[4] = {inner_upper[0] + 2, inner_upper[1] + 2, inner_upper[2] + 2, depth};
    auto res = std::make_shared<mesh::DataBlockArray<TV, IFORM, DOF>>(
            p_data->getPointer(), ndims + 1,
            o_lower, o_upper,
            data::FAST_FIRST,
            i_lower, i_upper);
    res->update();

    return std::dynamic_pointer_cast<mesh::DataBlock>(res);


}


template<typename TV, mesh::MeshEntityType IFORM>
std::shared_ptr<mesh::DataBlock>
create_data_block_t1(std::shared_ptr<mesh::AttributeBase> const &item, boost::shared_ptr<SAMRAI::hier::PatchData> pd)
{
    std::shared_ptr<mesh::DataBlock> res(nullptr);

    switch (item->dof())
    {
        case 1:
            res = create_data_block_t2<TV, IFORM, 1>(item, pd);
            break;
        case 3:
            res = create_data_block_t2<TV, IFORM, 3>(item, pd);
            break;
        case 9:
            res = create_data_block_t2<TV, IFORM, 9>(item, pd);
            break;
        default:
            UNIMPLEMENTED;
    }
    return res;


};

template<typename TV>
std::shared_ptr<mesh::DataBlock>
create_data_block_t0(std::shared_ptr<mesh::AttributeBase> const &item, boost::shared_ptr<SAMRAI::hier::PatchData> pd)
{
    std::shared_ptr<mesh::DataBlock> res(nullptr);

    switch (item->entity_type())
    {
        case mesh::VERTEX:
            res = create_data_block_t1<TV, mesh::VERTEX>(item, pd);
            break;
        case mesh::EDGE:
            res = create_data_block_t1<TV, mesh::EDGE>(item, pd);
            break;
        case mesh::FACE:
            res = create_data_block_t1<TV, mesh::FACE>(item, pd);
            break;
        case mesh::VOLUME:
            res = create_data_block_t1<TV, mesh::VOLUME>(item, pd);
            break;
        default:
            RUNTIME_ERROR << " EntityType is not supported!" << std::endl;
            break;
    }
    return res;
};

std::shared_ptr<mesh::DataBlock>
create_data_block(std::shared_ptr<mesh::AttributeBase> const &item, boost::shared_ptr<SAMRAI::hier::PatchData> pd)
{
    std::shared_ptr<mesh::DataBlock> res(nullptr);
    if (item->value_type_info() == typeid(float)) { res = create_data_block_t0<float>(item, pd); }
    else if (item->value_type_info() == typeid(double)) { res = create_data_block_t0<double>(item, pd); }
    else if (item->value_type_info() == typeid(int)) { res = create_data_block_t0<int>(item, pd); }
//    else if (item->value_type_info() == typeid(long)) { attr_choice_form<long>(item, std::forward<Args>(args)...); }
    else { RUNTIME_ERROR << "Unsupported m_value_ type" << std::endl; }
    ASSERT(res != nullptr);
    return res;
}
}//namespace detail


void
SAMRAIWorker::move_to(std::shared_ptr<mesh::Worker> &w, SAMRAI::hier::Patch &patch)
{
    auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());

    ASSERT(pgeom != nullptr);
    const double *dx = pgeom->getDx();
    const double *xlo = pgeom->getXLower();
    const double *xhi = pgeom->getXUpper();

    index_type lo[3] = {
            patch.getBox().lower()[0],
            patch.getBox().lower()[1],
            patch.getBox().lower()[2]
    };
    index_type hi[3] = {
            patch.getBox().upper()[0],
            patch.getBox().upper()[1],
            patch.getBox().upper()[2]
    };

    std::shared_ptr<mesh::MeshBlock> m = std::make_shared<mesh::MeshBlock>(
            3, lo, hi, dx, xlo,
            static_cast<id_type>(patch.getBox().getGlobalId().getOwnerRank() * 10000 +
                                 patch.getBox().getGlobalId().getLocalId().getValue())
    );
    //m->deploy();

    for (auto &ob:w->get_chart()->attributes())
    {
        auto &attr = ob->attribute();
        if (attr == nullptr) { return; }
        auto db = detail::create_data_block(
                attr, patch.getPatchData(m_samrai_variables_.at(attr->id()), getDataContext()));

        ob->move_to(m, db);
    }
    w->move_to(m);
}

/**
 *
 * Set initial data for solution variables on patch interior.
 * This routine is called whenever a new patch is introduced to the
 * AMR patch hierarchy.  Note that the routine does nothing unless
 * we are at the initial time.  In all other cases, conservative
 * interpolation from coarser levels and copies from patches at the
 * same mesh resolution are sufficient to set data.
 *
 */
void SAMRAIWorker::initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time,
                                         const bool initial_time)
{
    if (initial_time)
    {
        move_to(m_worker_, patch);
        m_worker_->initialize(data_time);
    }


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

double SAMRAIWorker::computeStableDtOnPatch(SAMRAI::hier::Patch &patch, const bool initial_time,
                                            const double dt_time)
{
    auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());

    return pgeom->getDx()[0] / 2.0;
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
    move_to(m_worker_, patch);

    m_worker_->set_physical_boundary_conditions(time);

    m_worker_->next_time_step(time, dt);
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

}


void SAMRAIWorker::setPhysicalBoundaryConditions(
        SAMRAI::hier::Patch &patch,
        const double fill_time,
        const SAMRAI::hier::IntVector &
        ghost_width_to_fill)
{
    move_to(m_worker_, patch);

    m_worker_->set_physical_boundary_conditions(fill_time);

}

/*
 *************************************************************************
 *
 * Register VisIt data_block writer to write data_block to plot files that may
 * be postprocessed by the VisIt tool.
 *
 *************************************************************************
 */

void SAMRAIWorker::registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer)
{
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}


/*
 *************************************************************************
 *
 * Write SAMRAIWorker object state to specified stream.
 *
 *************************************************************************
 */

void SAMRAIWorker::printClassData(std::ostream &os) const
{
    os << "\nSAMRAIWorker::printClassData..." << std::endl;
    os << "SAMRAIWorker: this = " << (SAMRAIWorker *) this << std::endl;
    os << "m_name_ = " << m_name_ << std::endl;
    os << "d_grid_geometry = " << d_grid_geometry.get() << std::endl;

    os << std::endl;

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
    SAMRAITimeIntegrator(std::string const &s, std::shared_ptr<mesh::Worker> const &w = nullptr);

    ~SAMRAITimeIntegrator();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void load(data::DataEntityTable const &);

    virtual void save(data::DataEntityTable *) const;


    virtual void deploy();

    virtual void tear_down();

    virtual bool is_valid() const;

    virtual size_type step() const;

    virtual bool remaining_steps() const;

    virtual Real time_now() const;

    virtual size_type next_step(Real dt_now);

    virtual void check_point();


private:
    bool m_is_valid_ = false;
    Real m_dt_now_ = 10000;

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
        : base_type(s, w)
{
    /*
      * Initialize SAMRAI::tbox::MPI.
      */
    SAMRAI::tbox::SAMRAI_MPI::init(GLOBAL_COMM.comm());

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
}

std::ostream &SAMRAITimeIntegrator::print(std::ostream &os, int indent) const
{
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(os);
    if (samrai_cfg != nullptr) samrai_cfg->printClassData(os);
    if (hyp_level_integrator != nullptr) hyp_level_integrator->printClassData(os);
    return os;
};


void SAMRAITimeIntegrator::load(data::DataEntityTable const &db) { UNIMPLEMENTED; }

void SAMRAITimeIntegrator::save(data::DataEntityTable *) const { UNIMPLEMENTED; }

namespace detail
{
void convert_database_r(data::DataEntity const &src, boost::shared_ptr<SAMRAI::tbox::Database> &dest,
                        std::string const &key = "")
{

    if (src.is_table())
    {
        auto sub_db = key == "" ? dest : dest->putDatabase(key);

        src.as_table().foreach(
                [&](std::string const &k, data::DataEntity const &v) { convert_database_r(v.as_table(), sub_db, k); });
    } else if (key == "") { return; }
    else if (src.is_null()) { dest->putDatabase(key); }
    else if (src.as_light().any().is_boolean()) { dest->putBool(key, src.as<bool>()); }
    else if (src.as_light().any().is_string()) { dest->putString(key, src.as<std::string>()); }
    else if (src.as_light().any().is_floating_point()) { dest->putDouble(key, src.as<double>()); }
    else if (src.as_light().any().is_integral()) { dest->putInteger(key, src.as<int>()); }
    else if (src.as_light().any().type() == typeid(nTuple<bool, 3>))
    {
        dest->putBoolArray(key, &src.as<nTuple<bool, 3 >>()[0], 3);
    } else if (src.as_light().any().type() == typeid(nTuple<int, 3>))
    {
        dest->putIntegerArray(key, &src.as<nTuple<int, 3 >>()[0], 3);
    } else if (src.as_light().any().type() == typeid(nTuple<double, 3>))
    {
        dest->putDoubleArray(key, &src.as<nTuple<double, 3 >>()[0], 3);
    }
//    else if (src.type() == typeid(box_type)) { dest->putDoubleArray(key, &src.as<box_type>()[0], 3); }
    else if (src.as_light().any().type() == typeid(index_box_type))
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
convert_database(data::DataEntityTable const &src, std::string const &s_name = "")
{
    auto dest = boost::dynamic_pointer_cast<SAMRAI::tbox::Database>(
            boost::make_shared<SAMRAI::tbox::MemoryDatabase>(s_name));
    convert_database_r(src, dest);
    return dest;
}
}//namespace detail{
void SAMRAITimeIntegrator::deploy()
{
    if (concept::Deployable::is_deployed()) { return; }
    concept::Deployable::deploy();

    bool use_refined_timestepping = db.get_value("use_refined_timestepping", true);

    SAMRAI::tbox::Dimension dim(ndims);

    samrai_cfg = detail::convert_database(db, name());



    /**
    * Create major algorithm and data objects which comprise application.
    * Each object will be initialized either from input data or restart
    * files, or a combination of both.  Refer to each class constructor
    * for details.  For more information on the composition of objects
    * for this application, see comments at top of file.
    */


    grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(
            dim, "CartesianGeometry", samrai_cfg->getDatabase("CartesianGeometry"));
//    grid_geometry->printClassData(std::cout);
    //---------------------------------

    patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>(
            "PatchHierarchy", grid_geometry, samrai_cfg->getDatabase("PatchHierarchy"));
//    patch_hierarchy->recursivePrint(std::cout, "", 1);
    //---------------------------------
    /***
     *  create hyp_level_integrator and error_detector
     */
    ASSERT(worker()->is_deployed());
    patch_worker = boost::make_shared<SAMRAIWorker>(worker(), dim, grid_geometry);

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
            db.get_value("output_writer_name", name() + " VisIt Writer"),
            db.get_value("output_dir_name", name()),
            db.get_value("visit_number_procs_per_file", int(1))
    );

    patch_worker->registerVisItDataWriter(visit_data_writer);


    m_dt_now_ = time_integrator->initializeHierarchy();
    m_is_valid_ = true;
    samrai_cfg->printClassData(std::cout);

    MESSAGE << name() << " is deployed!" << std::endl;
//    time_integrator->printClassData(std::cout);

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

size_type SAMRAITimeIntegrator::next_step(Real dt)
{
    assert(is_valid());

    MESSAGE << " Time = " << time_now() << " Step = " << step() << std::endl;
    Real loop_time = time_integrator->getIntegratorTime();
    Real loop_time_end = loop_time + dt;

    dt = std::min(dt, m_dt_now_);

    while (loop_time < loop_time_end && dt > 0 && time_integrator->stepsRemaining() > 0)
    {
        Real dt_new = time_integrator->advanceHierarchy(dt, false);
        loop_time += dt;
        dt = std::min(dt_new, loop_time_end - loop_time);
    }
}

void SAMRAITimeIntegrator::check_point()
{
    if (visit_data_writer != nullptr)
    {

        visit_data_writer->writePlotData(patch_hierarchy,
                                         time_integrator->getIntegratorStep(),
                                         time_integrator->getIntegratorTime());
    }
}

Real SAMRAITimeIntegrator::time_now() const { return static_cast<Real>( time_integrator->getIntegratorTime()); }

size_type
SAMRAITimeIntegrator::step() const { return static_cast<size_type>( time_integrator->getIntegratorStep()); }

bool
SAMRAITimeIntegrator::remaining_steps() const { return time_integrator->stepsRemaining(); }
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