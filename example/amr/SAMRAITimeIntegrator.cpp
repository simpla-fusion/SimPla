//
// Created by salmon on 16-10-24.
//

// Headers for SimPla
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include <simpla/algebra/nTupleExt.h>
#include <simpla/data/DataTable.h>
#include <simpla/engine/AttributeView.h>
#include <simpla/engine/DataBlock.h>
#include <simpla/engine/DomainView.h>
#include <simpla/engine/Patch.h>
#include <simpla/engine/TimeIntegrator.h>
#include <simpla/engine/Worker.h>
#include <simpla/engine/Worker.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/toolbox/Log.h>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <map>
#include <memory>
#include <string>
// Headers for SAMRAI
#include <SAMRAI/SAMRAI_config.h>

#include <SAMRAI/algs/HyperbolicLevelIntegrator.h>
#include <SAMRAI/algs/TimeRefinementIntegrator.h>
#include <SAMRAI/algs/TimeRefinementLevelStrategy.h>

#include <SAMRAI/mesh/BergerRigoutsos.h>
#include <SAMRAI/mesh/CascadePartitioner.h>
#include <SAMRAI/mesh/CascadePartitioner.h>
#include <SAMRAI/mesh/GriddingAlgorithm.h>
#include <SAMRAI/mesh/StandardTagAndInitialize.h>

#include <SAMRAI/hier/BoundaryBox.h>
#include <SAMRAI/hier/BoxContainer.h>
#include <SAMRAI/hier/Index.h>
#include <SAMRAI/hier/PatchDataRestartManager.h>
#include <SAMRAI/hier/PatchHierarchy.h>
#include <SAMRAI/hier/PatchLevel.h>
#include <SAMRAI/hier/VariableDatabase.h>

#include <SAMRAI/geom/CartesianGridGeometry.h>
#include <SAMRAI/geom/CartesianPatchGeometry.h>

#include <SAMRAI/pdat/CellData.h>
#include <SAMRAI/pdat/CellIndex.h>
#include <SAMRAI/pdat/CellIterator.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/CellVariable.h>
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/FaceIndex.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/NodeVariable.h>

#include <SAMRAI/tbox/BalancedDepthFirstTree.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <SAMRAI/tbox/MathUtilities.h>
#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/Utilities.h>

#include <SAMRAI/appu/BoundaryUtilityStrategy.h>
#include <SAMRAI/appu/CartesianBoundaryDefines.h>
#include <SAMRAI/appu/CartesianBoundaryDefines.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities2.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities3.h>
#include <SAMRAI/appu/VisItDataWriter.h>
#include <SAMRAI/pdat/NodeDoubleLinearTimeInterpolateOp.h>
#include <SAMRAI/pdat/SideVariable.h>
#include <simpla/engine/TimeIntegrator.h>
#include <simpla/physics/Constants.h>

namespace simpla {
struct SAMRAIPatchProxy : public engine::Patch {
   public:
    SAMRAIPatchProxy(SAMRAI::hier::Patch &patch, boost::shared_ptr<SAMRAI::hier::VariableContext> ctx,
                     std::map<id_type, std::shared_ptr<engine::AttributeDesc>> const &simpla_attrs_,
                     std::map<id_type, boost::shared_ptr<SAMRAI::hier::Variable>> const &samrai_variables);
    ~SAMRAIPatchProxy();
    std::shared_ptr<engine::DataBlock> data(id_type const &id, std::shared_ptr<engine::DataBlock> const &p = nullptr);
    std::shared_ptr<engine::DataBlock> data(id_type const &id) const;

   private:
    SAMRAI::hier::Patch &m_samrai_patch_;
    boost::shared_ptr<SAMRAI::hier::VariableContext> const &m_samrai_ctx_;
    std::map<id_type, std::shared_ptr<engine::AttributeDesc>> const &m_simpla_attrs_;
    std::map<id_type, boost::shared_ptr<SAMRAI::hier::Variable>> const &m_samrai_variables_;
};
SAMRAIPatchProxy::SAMRAIPatchProxy(SAMRAI::hier::Patch &patch, boost::shared_ptr<SAMRAI::hier::VariableContext> ctx,
                                   std::map<id_type, std::shared_ptr<engine::AttributeDesc>> const &simpla_attrs_,
                                   std::map<id_type, boost::shared_ptr<SAMRAI::hier::Variable>> const &var_map)
    : m_samrai_patch_(patch), m_samrai_ctx_(ctx), m_simpla_attrs_(simpla_attrs_), m_samrai_variables_(var_map) {
    auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());

    ASSERT(pgeom != nullptr);
    const double *dx = pgeom->getDx();
    const double *xlo = pgeom->getXLower();
    const double *xhi = pgeom->getXUpper();

    index_type lo[3] = {patch.getBox().lower()[0], patch.getBox().lower()[1], patch.getBox().lower()[2]};
    index_type hi[3] = {patch.getBox().upper()[0], patch.getBox().upper()[1], patch.getBox().upper()[2]};

    std::shared_ptr<simpla::engine::MeshBlock> m = std::make_shared<simpla::engine::MeshBlock>(3, lo, hi, dx, xlo);
    //    m->id(static_cast<id_type>(patch.getBox().getGlobalId().getOwnerRank() * 10000 +
    //                               patch.getBox().getGlobalId().getLocalId().getValue()));
//    this->SetMeshBlock(m);
};
SAMRAIPatchProxy::~SAMRAIPatchProxy() {}

namespace detail {

template <typename TV>
std::shared_ptr<engine::DataBlock> create_data_block_t0(engine::AttributeDesc const &desc,
                                                        boost::shared_ptr<SAMRAI::hier::PatchData> pd) {
    auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::NodeData<TV>>(pd);
    int ndims = p_data->getDim().getValue();
    int depth = p_data->getDepth();
    auto outer_lower = p_data->getGhostBox().lower();
    auto outer_upper = p_data->getGhostBox().upper();
    auto inner_lower = p_data->getBox().lower();
    auto inner_upper = p_data->getBox().upper();
    size_type dims[4] = {static_cast<size_type>(outer_upper[0] - outer_lower[0]),
                         static_cast<size_type>(outer_upper[1] - outer_lower[1]),
                         static_cast<size_type>(outer_upper[2] - outer_lower[2]), static_cast<size_type>(desc.GetDOF())};
    index_type lo[4] = {inner_lower[0] - outer_lower[0], inner_lower[1] - outer_lower[1],
                        inner_lower[2] - outer_lower[2], 0};
    index_type hi[4] = {inner_upper[0] - outer_lower[0], inner_upper[1] - outer_lower[1],
                        inner_upper[2] - outer_lower[2], desc.GetDOF()};
    auto res = std::make_shared<engine::DataBlockAdapter<Array<TV, 4>>>(p_data->getPointer(), dims, lo, hi);
    res->Initialize();
    return std::dynamic_pointer_cast<engine::DataBlock>(res);
};

std::shared_ptr<engine::DataBlock> create_data_block(engine::AttributeDesc const &desc,
                                                     boost::shared_ptr<SAMRAI::hier::PatchData> pd) {
    std::shared_ptr<engine::DataBlock> res(nullptr);
    if (desc.GetValueTypeInfo() == (typeid(float))) {
        res = create_data_block_t0<float>(desc, pd);
    } else if (desc.GetValueTypeInfo() == (typeid(double))) {
        res = create_data_block_t0<double>(desc, pd);
    } else if (desc.GetValueTypeInfo() == (typeid(int))) {
        res = create_data_block_t0<int>(desc, pd);
    }
    //    else if (item->GetValueTypeInfo() == typeid(long)) { attr_choice_form<long>(item,
    //    std::forward<Args>(args)...); }
    else {
        RUNTIME_ERROR << "Unsupported m_value_ type" << std::endl;
    }
    ASSERT(res != nullptr);
    return res;
}
}  // namespace detail
std::shared_ptr<engine::DataBlock> SAMRAIPatchProxy::data(id_type const &id,
                                                          std::shared_ptr<engine::DataBlock> const &p) {
    UNIMPLEMENTED;
}
std::shared_ptr<engine::DataBlock> SAMRAIPatchProxy::data(id_type const &id) const {
    return simpla::detail::create_data_block(*m_simpla_attrs_.at(id),
                                             m_samrai_patch_.getPatchData(m_samrai_variables_.at(id), m_samrai_ctx_));
}

class SAMRAI_HyperbolicPatchStrategyAdapter : public SAMRAI::algs::HyperbolicPatchStrategy {
   public:
    SAMRAI_HyperbolicPatchStrategyAdapter(engine::Manager *);

    /**
     * The destructor for SAMRAIWorkerHyperbolic does nothing.
     */
    ~SAMRAI_HyperbolicPatchStrategyAdapter();

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

    void registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator) final;
    void setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                           SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm) final;

    void initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time, const bool initial_time) final;
    double computeStableDtOnPatch(SAMRAI::hier::Patch &patch, const bool initial_time, const double dt_time) final;
    void computeFluxesOnPatch(SAMRAI::hier::Patch &patch, const double time, const double dt) final;

    /**
     * Update linear advection solution variables by performing a conservative
     * difference with the fluxes calculated in computeFluxesOnPatch().
     */
    void conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, const double time, const double dt,
                                       bool at_syncronization);

    /**
     * Tag cells for refinement using gradient detector.
     */
    void tagGradientDetectorCells(SAMRAI::hier::Patch &patch, const double regrid_time, const bool initial_error,
                                  const int tag_indexindx, const bool uses_richardson_extrapolation_too);

    /**
     * Tag cells for refinement using Richardson extrapolation.
     */
    void tagRichardsonExtrapolationCells(SAMRAI::hier::Patch &patch, const int error_level_number,
                                         const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
                                         const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse,
                                         const double regrid_time, const double deltat, const int error_coarsen_ratio,
                                         const bool initial_error, const int tag_index,
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
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch &patch, const double fill_time,
                                       const SAMRAI::hier::IntVector &ghost_width_to_fill);

    SAMRAI::hier::IntVector getRefineOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessRefine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse,
                          const SAMRAI::hier::Box &fine_box, const SAMRAI::hier::IntVector &ratio) {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    void postprocessRefine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse,
                           const SAMRAI::hier::Box &fine_box, const SAMRAI::hier::IntVector &ratio) {
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

    SAMRAI::hier::IntVector getCoarsenOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessCoarsen(SAMRAI::hier::Patch &coarse, const SAMRAI::hier::Patch &fine,
                           const SAMRAI::hier::Box &coarse_box, const SAMRAI::hier::IntVector &ratio) {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    void postprocessCoarsen(SAMRAI::hier::Patch &coarse, const SAMRAI::hier::Patch &fine,
                            const SAMRAI::hier::Box &coarse_box, const SAMRAI::hier::IntVector &ratio) {
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

    /**Print all data members for SAMRAIWorkerHyperbolic class.     */
    void printClassData(std::ostream &os) const;

    //    void Dispatch(SAMRAI::hier::Patch &patch);

   private:
    std::shared_ptr<engine::DomainView> m_domain_view_;
    /*
     * The object GetName is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string m_name_;
    SAMRAI::tbox::Dimension d_dim;

    /*
     * We cache pointers to the grid geometry object to set up initial
     * GetDataBlock, Set physical boundary conditions, and register plot
     * variables.
     */
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> d_grid_geometry = nullptr;
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer = nullptr;

    /*
     * Data items used for nonuniform Load balance, if used.
     */
    boost::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_workload_variable;
    int d_workload_data_id;
    bool d_use_nonuniform_workload;
    std::map<id_type, boost::shared_ptr<SAMRAI::hier::Variable>> m_samrai_variables_;
    //    boost::shared_ptr<SAMRAI::pdat::NodeVariable<double>> d_xyz;
    engine::Manager *m_manager_ = nullptr;
    SAMRAI::hier::IntVector d_nghosts;
    SAMRAI::hier::IntVector d_fluxghosts;
};

SAMRAI_HyperbolicPatchStrategyAdapter::SAMRAI_HyperbolicPatchStrategyAdapter(engine::Manager *m)
    : SAMRAI::algs::HyperbolicPatchStrategy(),
      d_dim(4),
      m_manager_(m),
      //      d_grid_geometry(grid_geom),
      d_use_nonuniform_workload(false),
      d_nghosts(d_dim, 4),
      d_fluxghosts(d_dim, 1) {
    //    TBOX_ASSERT(grid_geom);
}

/*
 *************************************************************************
 *
 * Empty destructor for SAMRAIWorkerAdapter class.
 *
 *************************************************************************
 */

SAMRAI_HyperbolicPatchStrategyAdapter::~SAMRAI_HyperbolicPatchStrategyAdapter() {}

// void SAMRAI_HyperbolicPatchStrategyAdapter::Dispatch(SAMRAI::hier::Patch &patch) {
//    engine::DomainView::Dispatch(std::dynamic_pointer_cast<engine::Patch>(
//        std::make_shared<SAMRAIPatchProxy>(patch, getDataContext(), this->GetAttributeDict(), m_samrai_variables_)));
//}

namespace detail {
template <typename T>
boost::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable_t(int ndims, engine::AttributeDesc const &attr) {
    static int var_depth[4] = {1, 3, 3, 1};
    SAMRAI::tbox::Dimension d_dim(ndims);
    return (attr.GetIFORM() > VOLUME)
               ? nullptr
               : boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(boost::make_shared<SAMRAI::pdat::NodeVariable<T>>(
                     d_dim, attr.GetName(), var_depth[attr.GetIFORM()] * attr.GetDOF()));
}

boost::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable(unsigned int ndims,
                                                                 engine::AttributeDesc const &desc) {
    if (desc.GetValueTypeInfo() == (typeid(float))) {
        return create_samrai_variable_t<float>(ndims, desc);
    } else if (desc.GetValueTypeInfo() == (typeid(double))) {
        return create_samrai_variable_t<double>(ndims, desc);
    } else if (desc.GetValueTypeInfo() == (typeid(int))) {
        return create_samrai_variable_t<int>(ndims, desc);
    }
    //    else if (item->GetValueTypeInfo() == typeid(long)) { attr_choice_form<long>(item,
    //    std::forward<Args>(args)...); }
    else {
        RUNTIME_ERROR << " value type [" << desc.GetValueTypeInfo().name() << "] is not supported!" << std::endl;
    }
    return nullptr;
}
}  // namespace detail{
   /**
    * Register conserved variables  and  register plot data with VisIt.
    */

void SAMRAI_HyperbolicPatchStrategyAdapter::registerModelVariables(
    SAMRAI::algs::HyperbolicLevelIntegrator *integrator) {
    ASSERT(integrator != nullptr);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

    if (!d_visit_writer) {
        RUNTIME_ERROR << m_name_ << ": registerModelVariables() VisIt GetDataBlock writer was not registered."
                                    "Consequently, no plot GetDataBlock will be written."
                      << std::endl;
    }
    //    SAMRAI::tbox::Dimension d_dim{4};
    SAMRAI::hier::IntVector d_nghosts{d_dim, 4};
    SAMRAI::hier::IntVector d_fluxghosts{d_dim, 1};
    //**************************************************************
//    for (auto const &item : m_domain_view_->GetAttributeDict()) {
//        boost::shared_ptr<SAMRAI::hier::Variable> var = simpla::detail::create_samrai_variable(3, *item.second);
//        m_samrai_variables_[item.second->GetGUID()] = var;
//
//        engine::AttributeDesc const &attr = *item.second;
//        data::DataTable &attr_db = m_domain_view_->attr_db(attr.GetGUID());
//        //                static const char visit_variable_type[3][10] = {"SCALAR", "VECTOR",
//        //                "TENSOR"};
//        //                static const char visit_variable_type2[4][10] = {"SCALAR", "VECTOR",
//        //                "VECTOR", "SCALAR"};
//
//        /*** FIXME:
//        *  1. SAMRAI Visit Writer only support NODE and CELL variable (double,float ,int)
//        *  2. SAMRAI   SAMRAI::algs::HyperbolicLevelIntegrator->registerVariable only support double
//        **/
//
//        if (attr_db.check("COORDINATES", true)) {
//            VERBOSE << attr.GetName() << " is registered as coordinate" << std::endl;
//            integrator->registerVariable(var, d_nghosts, SAMRAI::algs::HyperbolicLevelIntegrator::INPUT,
//                                         d_grid_geometry, "", "LINEAR_REFINE");
//
//        } else if (attr_db.check("FLUX", true)) {
//            integrator->registerVariable(var, d_fluxghosts, SAMRAI::algs::HyperbolicLevelIntegrator::FLUX,
//                                         d_grid_geometry, "CONSERVATIVE_COARSEN", "NO_REFINE");
//
//        } else if (attr_db.check("INPUT", true)) {
//            integrator->registerVariable(var, d_nghosts, SAMRAI::algs::HyperbolicLevelIntegrator::INPUT,
//                                         d_grid_geometry, "", "NO_REFINE");
//        } else {
//            switch (attr.GetIFORM()) {
//                case EDGE:
//                case FACE:
//                //                    integrator->registerVariable(var, d_nghosts,
//                //                    SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
//                //                                                 d_grid_geometry, "CONSERVATIVE_COARSEN",
//                //                                                 "CONSERVATIVE_LINEAR_REFINE");
//                //                    break;
//                case VERTEX:
//                case VOLUME:
//                default:
//                    integrator->registerVariable(var, d_nghosts, SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP,
//                                                 d_grid_geometry, "", "LINEAR_REFINE");
//            }
//
//            //            VERBOSE << (attr.GetName()) << " --  " << visit_variable_type << std::endl;
//        }
//
//        std::string visit_variable_type = "";
//        if ((attr.GetIFORM() == VERTEX || attr.GetIFORM() == VOLUME) && attr.GetDOF() == 1) {
//            visit_variable_type = "SCALAR";
//        } else if (((attr.GetIFORM() == EDGE || attr.GetIFORM() == FACE) && attr.GetDOF() == 1) ||
//                   ((attr.GetIFORM() == VERTEX || attr.GetIFORM() == VOLUME) && attr.GetDOF() == 3)) {
//            visit_variable_type = "VECTOR";
//        } else if (((attr.GetIFORM() == VERTEX || attr.GetIFORM() == VOLUME) && attr.GetDOF() == 9) ||
//                   ((attr.GetIFORM() == EDGE || attr.GetIFORM() == FACE) && attr.GetDOF() == 3)) {
//            visit_variable_type = "TENSOR";
//        } else {
//            WARNING << "Can not register attribute [" << attr.GetName() << "] to VisIt  writer!" << std::endl;
//        }
//
//        if (visit_variable_type != "" && attr_db.check("CHECK", true)) {
//            d_visit_writer->registerPlotQuantity(
//                attr.GetName(), visit_variable_type,
//                vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
//
//        } else if (attr_db.check("COORDINATES", true)) {
//            d_visit_writer->registerNodeCoordinates(
//                vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
//        }
//    };
    //    integrator->printClassData(std::cout);
    //    vardb->printClassData(std::cout);
}

/**
 * Set up parameters for nonuniform load balancing, if used.
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                                                              SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm) {
    const SAMRAI::hier::IntVector &zero_vec = SAMRAI::hier::IntVector::getZero(d_dim);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();
    if (d_use_nonuniform_workload && gridding_algorithm) {
        auto load_balancer =
            boost::dynamic_pointer_cast<SAMRAI::mesh::CascadePartitioner>(gridding_algorithm->getLoadBalanceStrategy());
        if (load_balancer) {
            d_workload_variable.reset(new SAMRAI::pdat::CellVariable<double>(d_dim, "workload_variable", 1));
            d_workload_data_id =
                vardb->registerVariableAndContext(d_workload_variable, vardb->getContext("WORKLOAD"), zero_vec);
            load_balancer->setWorkloadPatchDataIndex(d_workload_data_id);
        } else {
            WARNING << m_name_ << ": "
                    << "  Unknown Load balancer used in gridding algorithm."
                    << "  Ignoring request for nonuniform Load balancing." << std::endl;
            d_use_nonuniform_workload = false;
        }
    } else {
        d_use_nonuniform_workload = false;
    }
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

void SAMRAI_HyperbolicPatchStrategyAdapter::initializeDataOnPatch(SAMRAI::hier::Patch &patch, const double data_time,
                                                                  const bool initial_time) {
    // FIXME:
    //    Dispatch(patch);

    if (initial_time) {}

    if (d_use_nonuniform_workload) {
        if (!patch.checkAllocated(d_workload_data_id)) { patch.allocatePatchData(d_workload_data_id); }
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

double SAMRAI_HyperbolicPatchStrategyAdapter::computeStableDtOnPatch(SAMRAI::hier::Patch &patch,
                                                                     const bool initial_time, const double dt_time) {
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

void SAMRAI_HyperbolicPatchStrategyAdapter::computeFluxesOnPatch(SAMRAI::hier::Patch &patch, const double time,
                                                                 const double dt) {}

/*
 *************************************************************************
 *
 * Update solution variables by performing a conservative
 * difference with the fluxes calculated in computeFluxesOnPatch().
 *
 *************************************************************************
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, const double time,
                                                                          const double dt, bool at_syncronization) {
    // FIXME: Dispatch(patch);
    //    this->SetPhysicalBoundaryConditions(time);
}

/*
 *************************************************************************
 *
 * Tag cells for refinement using Richardson extrapolation.  Criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::tagRichardsonExtrapolationCells(
    SAMRAI::hier::Patch &patch, const int error_level_number,
    const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
    const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse, const double regrid_time,
    const double deltat, const int error_coarsen_ratio, const bool initial_error, const int tag_index,
    const bool uses_gradient_detector_too) {}

/*
 *************************************************************************
 *
 * Tag cells for refinement using gradient detector.  Tagging criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::tagGradientDetectorCells(SAMRAI::hier::Patch &patch,
                                                                     const double regrid_time, const bool initial_error,
                                                                     const int tag_indx,
                                                                     const bool uses_richardson_extrapolation_too) {}

void SAMRAI_HyperbolicPatchStrategyAdapter::setPhysicalBoundaryConditions(
    SAMRAI::hier::Patch &patch, const double fill_time, const SAMRAI::hier::IntVector &ghost_width_to_fill) {
    // FIXME:    Dispatch(patch);
    //    this->SetPhysicalBoundaryConditions(fill_time);
}

/*
 *************************************************************************
 *
 * Register VisIt SetDataBlock writer to write GetDataBlock to plot files that may
 * be postprocessed by the VisIt tool.
 *
 *************************************************************************
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::registerVisItDataWriter(
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer) {
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}

/*
 *************************************************************************
 *
 * Write SAMRAIWorkerAdapter object state to specified stream.
 *
 *************************************************************************
 */

void SAMRAI_HyperbolicPatchStrategyAdapter::printClassData(std::ostream &os) const {
    os << "\nSAMRAIWorkerAdapter::printClassData..." << std::endl;
    os << "m_name_ = " << m_name_ << std::endl;
    os << "d_grid_geometry = " << d_grid_geometry.get() << std::endl;
    os << std::endl;
}

struct SAMRAITimeIntegrator : public engine::TimeIntegrator {
    typedef engine::TimeIntegrator base_type;

   public:
    SAMRAITimeIntegrator();
    ~SAMRAITimeIntegrator();
    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    virtual void Update();
    virtual void Finalize();
    virtual size_type step() const;
    virtual bool remainingSteps() const;
    virtual Real timeNow() const;
    virtual size_type NextTimeStep(Real dt_now);
    virtual void CheckPoint();

   private:
    bool m_is_valid_ = false;
    Real m_dt_now_ = 10000;

    boost::shared_ptr<SAMRAI::tbox::Database> samrai_cfg;
    boost::shared_ptr<SAMRAI_HyperbolicPatchStrategyAdapter> hyperbolic_patch_strategy;
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;
    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;
    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

    //    boost::shared_ptr<SAMRAI::engine::StandardTagAndInitialize> error_detector;
    //    boost::shared_ptr<SAMRAI::engine::BergerRigoutsos> box_generator;
    //    boost::shared_ptr<SAMRAI::engine::CascadePartitioner> load_balancer;
    //    boost::shared_ptr<SAMRAI::engine::GriddingAlgorithm> gridding_algorithm;
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
std::shared_ptr<engine::TimeIntegrator> create_time_integrator() {
    return std::dynamic_pointer_cast<engine::TimeIntegrator>(std::make_shared<SAMRAITimeIntegrator>());
}

SAMRAITimeIntegrator::SAMRAITimeIntegrator() : base_type() {
    /** Setup SAMRAI::tbox::MPI.      */
    SAMRAI::tbox::SAMRAI_MPI::init(GLOBAL_COMM.comm());
    SAMRAI::tbox::SAMRAIManager::initialize();
    /** Setup SAMRAI, enable logging, and process command line.     */
    SAMRAI::tbox::SAMRAIManager::startup();
    //    const SAMRAI::tbox::SAMRAI_MPI & mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

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
    GriddingAlgorithm{}

    // Refer to mesh::BergerRigoutsos for input
    BergerRigoutsos {
       sort_output_nodes = TRUE // Makes results repeatable.
       efficiency_tolerance   = 0.85e0    // min % of tag cells in new patch level
       combine_efficiency     = 0.95e0    // chop box if sum of volumes of smaller
                                          // boxes < efficiency * vol of large box
    }
    // Refer to mesh::StandardTagAndInitialize for input
    StandardTagAndInitialize {tagging_method = "GRADIENT_DETECTOR"}
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

    db().SetValue("CartesianGeometry.domain_boxes_0", index_box_type{{0, 0, 0}, {16, 16, 16}});
    db().SetValue("CartesianGeometry.periodic_dimension", nTuple<int, 3>{1, 1, 1});
    db().SetValue("CartesianGeometry.x_lo", nTuple<double, 3>{1.0, 0.0, -1.0});
    db().SetValue("CartesianGeometry.x_up", nTuple<double, 3>{2, PI, 1});
    // Maximum number of levels in hierarchy.
    db().SetValue("PatchHierarchy.max_levels", int(3));
    db().SetValue("PatchHierarchy.ratio_to_coarser.level_1", nTuple<int, 3>{2, 2, 1});
    db().SetValue("PatchHierarchy.ratio_to_coarser.level_2", nTuple<int, 3>{2, 2, 1});
    db().SetValue("PatchHierarchy.ratio_to_coarser.level_3", nTuple<int, 3>{2, 2, 1});
    db().SetValue("PatchHierarchy.largest_patch_size.level_0", nTuple<int, 3>{32, 32, 32});
    db().SetValue("PatchHierarchy.smallest_patch_size.level_0", nTuple<int, 3>{4, 4, 4});

    db().CreateTable("GriddingAlgorithm");
    // Makes results repeatable.
    db().SetValue("BergerRigoutsos.sort_output_nodes", true);
    // min % of GetTag cells in new patch level
    db().SetValue("BergerRigoutsos.efficiency_tolerance", 0.85);
    // chop box if sum of volumes of smaller
    //    // boxes < efficiency * vol of large box
    db().SetValue("BergerRigoutsos.combine_efficiency", 0.95);

    // Refer to mesh::StandardTagAndInitialize for input
    db().SetValue("StandardTagAndInitialize.tagging_method", "GRADIENT_DETECTOR");

    // Refer to algs::HyperbolicLevelIntegrator for input
    // max cfl factor used in problem
    db().SetValue("HyperbolicLevelIntegrator.cfl", 0.9);
    db().SetValue("HyperbolicLevelIntegrator.cfl_init", 0.9);  // initial cfl factor
    db().SetValue("HyperbolicLevelIntegrator.lag_dt_computation", true);
    db().SetValue("HyperbolicLevelIntegrator.use_ghosts_to_compute_dt", true);

    // Refer to algs::TimeRefinementIntegrator for input
    // initial simulation time
    db().SetValue("TimeRefinementIntegrator.start_time", 0.e0);
    // final simulation time
    db().SetValue("TimeRefinementIntegrator.end_time", 1.e0);
    // growth factor for timesteps
    db().SetValue("TimeRefinementIntegrator.grow_dt", 1.1e0);
    // max number of simulation timesteps
    db().SetValue("TimeRefinementIntegrator.max_integrator_steps", 5);

    // Refer to mesh::TreeLoadBalancer for input
    db().CreateTable("LoadBalancer");
}

SAMRAITimeIntegrator::~SAMRAITimeIntegrator() {
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
}

std::ostream &SAMRAITimeIntegrator::Print(std::ostream &os, int indent) const {
    SAMRAI::hier::VariableDatabase::getDatabase()->printClassData(os);
    if (samrai_cfg != nullptr) samrai_cfg->printClassData(os);
    if (hyp_level_integrator != nullptr) hyp_level_integrator->printClassData(os);
    return os;
};

namespace detail {
void convert_database_r(data::DataEntity const &src, boost::shared_ptr<SAMRAI::tbox::Database> &dest,
                        std::string const &key = "") {
    if (src.isTable()) {
        auto sub_db = key == "" ? dest : dest->putDatabase(key);
        src.asTable().foreach (
            [&](std::string const &k, data::DataEntity const &v) { convert_database_r(v, sub_db, k); });
    } else if (key == "") {
        return;
    } else if (src.isNull()) {
        dest->putDatabase(key);
    } else if (src.asLight().isBoolean()) {
        dest->putBool(key, src.asLight().as<bool>());
    } else if (src.asLight().is_string()) {
        dest->putString(key, src.asLight().as<std::string>());
    } else if (src.asLight().isFloatingPoint()) {
        dest->putDouble(key, src.asLight().as<double>());
    } else if (src.asLight().isIntegral()) {
        dest->putInteger(key, src.asLight().as<int>());
    } else if (src.asLight().type() == typeid(nTuple<bool, 3>)) {
        dest->putBoolArray(key, &src.asLight().as<nTuple<bool, 3>>()[0], 3);
    } else if (src.asLight().type() == typeid(nTuple<int, 3>)) {
        dest->putIntegerArray(key, &src.asLight().as<nTuple<int, 3>>()[0], 3);
    } else if (src.asLight().type() == typeid(nTuple<double, 3>)) {
        dest->putDoubleArray(key, &src.asLight().as<nTuple<double, 3>>()[0], 3);
    }
    //    else if (src.type() == typeid(box_type)) { dest->putDoubleArray(key,
    //    &src.as<box_type>()[0], 3); }
    else if (src.asLight().type() == typeid(index_box_type)) {
        nTuple<int, 3> i_lo, i_up;
        std::tie(i_lo, i_up) = src.asLight().as<index_box_type>();
        SAMRAI::tbox::Dimension dim(3);
        dest->putDatabaseBox(key, SAMRAI::tbox::DatabaseBox(dim, &(i_lo[0]), &(i_up[0])));
    } else {
        WARNING << " Unknown type [" << src << "]" << std::endl;
    }
}

boost::shared_ptr<SAMRAI::tbox::Database> convert_database(data::DataTable const &src, std::string const &s_name = "") {
    auto dest =
        boost::dynamic_pointer_cast<SAMRAI::tbox::Database>(boost::make_shared<SAMRAI::tbox::MemoryDatabase>(s_name));
    convert_database_r(src, dest);
    return dest;
}
}  // namespace detail{

void SAMRAITimeIntegrator::Update() {
    if (isUpdated()) { return; }
    engine::Manager::Update();
    bool use_refined_timestepping = db().GetValue("use_refined_timestepping", true);
    SAMRAI::tbox::Dimension dim(ndims);
    samrai_cfg = simpla::detail::convert_database(db(), name());
    samrai_cfg->printClassData(std::cout);
    /**
    * Create major algorithm and data objects which comprise application.
    * Each object will be initialized either from input data or restart
    * files, or a combination of both.  Refer to each class constructor
    * for details.  For more information on the composition of objects
    * for this application, see comments at top of file.
    */

    grid_geometry = boost::make_shared<SAMRAI::geom::CartesianGridGeometry>(
        dim, "CartesianGeometry", samrai_cfg->getDatabase("CartesianGeometry"));
    grid_geometry->printClassData(std::cout);
    //---------------------------------

    patch_hierarchy = boost::make_shared<SAMRAI::hier::PatchHierarchy>("PatchHierarchy", grid_geometry,
                                                                       samrai_cfg->getDatabase("PatchHierarchy"));
    //    patch_hierarchy->recursivePrint(std::cout, "", 1);
    //---------------------------------
    /***
     *  create hyp_level_integrator and error_detector
     */
    hyperbolic_patch_strategy = boost::make_shared<SAMRAI_HyperbolicPatchStrategyAdapter>(this);

    hyp_level_integrator = boost::make_shared<SAMRAI::algs::HyperbolicLevelIntegrator>(
        "SAMRAILevelIntegrator", samrai_cfg->getDatabase("HyperbolicLevelIntegrator"), hyperbolic_patch_strategy.get(),
        use_refined_timestepping);

    hyp_level_integrator->printClassData(std::cout);

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
        "StandardTagAndInitialize", hyp_level_integrator.get(), samrai_cfg->getDatabase("StandardTagAndInitialize"));
    //---------------------------------

    /**
     *  create grid_algorithm
     */
    auto box_generator =
        boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(dim, samrai_cfg->getDatabase("BergerRigoutsos"));

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(dim, "LoadBalancer",
                                                                              samrai_cfg->getDatabase("LoadBalancer"));

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    load_balancer->printStatistics(std::cout);

    auto gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
        patch_hierarchy, "GriddingAlgorithm", samrai_cfg->getDatabase("GriddingAlgorithm"), error_detector,
        box_generator, load_balancer);

    gridding_algorithm->printClassData(std::cout);
    //---------------------------------

    time_integrator = boost::make_shared<SAMRAI::algs::TimeRefinementIntegrator>(
        "TimeRefinementIntegrator", samrai_cfg->getDatabase("TimeRefinementIntegrator"), patch_hierarchy,
        hyp_level_integrator, gridding_algorithm);

    visit_data_writer = boost::make_shared<SAMRAI::appu::VisItDataWriter>(
        dim, db().GetValue("output_writer_name", name() + " VisIt Writer"), db().GetValue("output_dir_name", name()),
        db().GetValue("visit_number_procs_per_file", int(1)));

    hyperbolic_patch_strategy->registerVisItDataWriter(visit_data_writer);

    samrai_cfg->printClassData(std::cout);

    m_dt_now_ = time_integrator->initializeHierarchy();
    m_is_valid_ = true;

    MESSAGE << name() << " is deployed!" << std::endl;
    time_integrator->printClassData(std::cout);
};

void SAMRAITimeIntegrator::Finalize() {
    m_is_valid_ = false;
    visit_data_writer.reset();
    time_integrator.reset();
    hyp_level_integrator.reset();
    hyperbolic_patch_strategy.reset();
}

size_type SAMRAITimeIntegrator::NextTimeStep(Real dt) {
    MESSAGE << " Time = " << timeNow() << " Step = " << step() << std::endl;
    Real loop_time = time_integrator->getIntegratorTime();
    Real loop_time_end = loop_time + dt;

    dt = std::min(dt, m_dt_now_);
    while (loop_time < loop_time_end && dt > 0 && time_integrator->stepsRemaining() > 0) {
        Real dt_new = time_integrator->advanceHierarchy(dt, false);
        loop_time += dt;
        dt = std::min(dt_new, loop_time_end - loop_time);
    }
    return 0;
}

void SAMRAITimeIntegrator::CheckPoint() {
    if (visit_data_writer != nullptr) {
        visit_data_writer->writePlotData(patch_hierarchy, time_integrator->getIntegratorStep(),
                                         time_integrator->getIntegratorTime());
    }
}

Real SAMRAITimeIntegrator::timeNow() const { return static_cast<Real>(time_integrator->getIntegratorTime()); }
size_type SAMRAITimeIntegrator::step() const { return static_cast<size_type>(time_integrator->getIntegratorStep()); }
bool SAMRAITimeIntegrator::remainingSteps() const { return time_integrator->stepsRemaining(); }
}  // namespace simpla
