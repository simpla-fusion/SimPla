//
// Created by salmon on 16-10-24.
//
#include "SAMRAITimeIntegrator.h"
// Headers for SimPla
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include <simpla/engine/all.h>
#include <simpla/parallel/MPIComm.h>
#include <simpla/utilities/Log.h>
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
#include <SAMRAI/pdat/EdgeVariable.h>
#include <SAMRAI/pdat/FaceData.h>
#include <SAMRAI/pdat/FaceIndex.h>
#include <SAMRAI/pdat/FaceVariable.h>
#include <SAMRAI/pdat/NodeVariable.h>
#include <SAMRAI/pdat/SparseDataVariable.h>

#include <SAMRAI/tbox/BalancedDepthFirstTree.h>
#include <SAMRAI/tbox/Database.h>
#include <SAMRAI/tbox/InputDatabase.h>
#include <SAMRAI/tbox/InputManager.h>
#include <SAMRAI/tbox/MathUtilities.h>
#include <SAMRAI/tbox/PIO.h>
#include <SAMRAI/tbox/RestartManager.h>
#include <SAMRAI/tbox/SAMRAIManager.h>
#include <SAMRAI/tbox/SAMRAI_MPI.h>
#include <SAMRAI/tbox/Utilities.h>

#include <SAMRAI/appu/BoundaryUtilityStrategy.h>
#include <SAMRAI/appu/CartesianBoundaryDefines.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities2.h>
#include <SAMRAI/appu/CartesianBoundaryUtilities3.h>
#include <SAMRAI/appu/VisItDataWriter.h>
#include <SAMRAI/pdat/NodeDoubleLinearTimeInterpolateOp.h>
#include <SAMRAI/pdat/SideVariable.h>

namespace simpla {

REGISTER_CREATOR(SAMRAITimeIntegrator)

class SAMRAIHyperbolicPatchStrategyAdapter : public SAMRAI::algs::HyperbolicPatchStrategy {
    SP_OBJECT_BASE(SAMRAIHyperbolicPatchStrategyAdapter)
   public:
    SAMRAIHyperbolicPatchStrategyAdapter(std::shared_ptr<engine::Context> ctx,
                                         boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom);

    /**
     * The destructor for SAMRAIWorkerHyperbolic does nothing.
     */
    ~SAMRAIHyperbolicPatchStrategyAdapter() override;

    SP_DEFAULT_CONSTRUCT(SAMRAIHyperbolicPatchStrategyAdapter)
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

    void registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator) override;

    void setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                           SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm) override;

    void initializeDataOnPatch(SAMRAI::hier::Patch &patch, double data_time, bool initial_time) override;
    double computeStableDtOnPatch(SAMRAI::hier::Patch &patch, bool initial_time, double dt_time) override;
    void computeFluxesOnPatch(SAMRAI::hier::Patch &patch, double time, double dt) override;

    /**
     * Update linear advection solution variables by performing a conservative
     * difference with the fluxes calculated in computeFluxesOnPatch().
     */
    void conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, double time_now, double time_dt,
                                       bool at_syncronization) override;

    /**
     * Tag cells for refinement using gradient detector.
     */
    void tagGradientDetectorCells(SAMRAI::hier::Patch &patch, double regrid_time, bool initial_error, int tag_indx,
                                  bool uses_richardson_extrapolation_too) override;

    /**
     * Tag cells for refinement using Richardson extrapolation.
     */
    void tagRichardsonExtrapolationCells(SAMRAI::hier::Patch &patch, int error_level_number,
                                         const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
                                         const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse,
                                         double regrid_time, double deltat, int error_coarsen_ratio, bool initial_error,
                                         int tag_index, bool uses_gradient_detector_too) override;

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
    void setPhysicalBoundaryConditions(SAMRAI::hier::Patch &patch, double fill_time,
                                       const SAMRAI::hier::IntVector &ghost_width_to_fill) override;

    SAMRAI::hier::IntVector getRefineOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const override {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessRefine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse,
                          const SAMRAI::hier::Box &fine_box, const SAMRAI::hier::IntVector &ratio) override {
        NULL_USE(fine);
        NULL_USE(coarse);
        NULL_USE(fine_box);
        NULL_USE(ratio);
    }

    void postprocessRefine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse,
                           const SAMRAI::hier::Box &fine_box, const SAMRAI::hier::IntVector &ratio) override {
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

    SAMRAI::hier::IntVector getCoarsenOpStencilWidth(const SAMRAI::tbox::Dimension &dim) const override {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    void preprocessCoarsen(SAMRAI::hier::Patch &coarse, const SAMRAI::hier::Patch &fine,
                           const SAMRAI::hier::Box &coarse_box, const SAMRAI::hier::IntVector &ratio) override {
        NULL_USE(coarse);
        NULL_USE(fine);
        NULL_USE(coarse_box);
        NULL_USE(ratio);
    }

    void postprocessCoarsen(SAMRAI::hier::Patch &coarse, const SAMRAI::hier::Patch &fine,
                            const SAMRAI::hier::Box &coarse_box, const SAMRAI::hier::IntVector &ratio) override {
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
    static constexpr int NDIMS = 3;
    std::shared_ptr<engine::Context> m_ctx_;
    /*
     * The object GetPrefix is used for error/warning reporting and also as a
     * string label for restart database entries.
     */
    std::string m_name_;
    SAMRAI::tbox::Dimension d_dim;

    /*
     * We cache pointers to the grid geometry object to set up initial
     * GetDataBlock, SetValue physical boundary conditions, and register plot
     * variables.
     */
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> d_grid_geometry = nullptr;
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer = nullptr;

    /*
     * Data items used for nonuniform Load balance, if used.
     */
    boost::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_workload_variable;
    int d_workload_data_id = 0;
    bool d_use_nonuniform_workload;
    std::map<std::shared_ptr<engine::AttributeDesc>, boost::shared_ptr<SAMRAI::hier::Variable>> m_samrai_variables_;
    SAMRAI::hier::IntVector d_nghosts;
    SAMRAI::hier::IntVector d_fluxghosts;

    void ConvertPatchFromSAMRAI(SAMRAI::hier::Patch &patch, engine::Patch *p);
    void ConvertPatchToSAMRAI(SAMRAI::hier::Patch &patch, engine::Patch *p);
};

SAMRAIHyperbolicPatchStrategyAdapter::SAMRAIHyperbolicPatchStrategyAdapter(
    std::shared_ptr<engine::Context> ctx, boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom)
    : d_dim(3),
      d_grid_geometry(grid_geom),
      d_use_nonuniform_workload(false),
      d_nghosts(d_dim, 4),
      d_fluxghosts(d_dim, 1),
      m_ctx_(std::move(ctx)) {
    TBOX_ASSERT(grid_geom);
}

/**************************************************************************
 *
 * Empty destructor for SAMRAIWorkerAdapter class.
 *
 **************************************************************************/

SAMRAIHyperbolicPatchStrategyAdapter::~SAMRAIHyperbolicPatchStrategyAdapter() = default;

namespace detail {
template <typename T>
boost::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable_t(const std::shared_ptr<engine::AttributeDesc> attr) {
    SAMRAI::tbox::Dimension d_dim(3);

    boost::shared_ptr<SAMRAI::hier::Variable> res;
    switch (attr->GetIFORM()) {
        case VERTEX:
            res = boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
                boost::make_shared<SAMRAI::pdat::NodeVariable<T>>(d_dim, attr->GetPrefix(), attr->GetDOF()));
            break;
        case EDGE:
            res = boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
                boost::make_shared<SAMRAI::pdat::EdgeVariable<T>>(d_dim, attr->GetPrefix(), attr->GetDOF()));
            break;
        case FACE:
            res = boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
                boost::make_shared<SAMRAI::pdat::SideVariable<T>>(d_dim, attr->GetPrefix(), attr->GetDOF()));
            break;
        case VOLUME:
            res = boost::dynamic_pointer_cast<SAMRAI::hier::Variable>(
                boost::make_shared<SAMRAI::pdat::CellVariable<T>>(d_dim, attr->GetPrefix(), attr->GetDOF()));
            break;
        default:
            break;
    }
    return res;
}

boost::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable(const std::shared_ptr<engine::AttributeDesc> &attr) {
    boost::shared_ptr<SAMRAI::hier::Variable> res = nullptr;

    if (attr->value_type_info() == (typeid(float))) {
        res = create_samrai_variable_t<float>(attr);
    } else if (attr->value_type_info() == (typeid(double))) {
        res = create_samrai_variable_t<double>(attr);
    } else if (attr->value_type_info() == (typeid(int))) {
        res = create_samrai_variable_t<int>(attr);
    } else {
        RUNTIME_ERROR << " attr [" << attr->GetPrefix() << "] is not supported!" << std::endl;
    }
    return res;
}

template <typename T, int NDIMS>
Array<T, NDIMS> create_array(SAMRAI::pdat::ArrayData<T> &p_data, int depth = 0) {
    auto i_lower = p_data.getBox().lower();
    auto i_upper = p_data.getBox().upper();

    typename Array<T, NDIMS>::array_index_box_type i_box{{i_lower[0], i_lower[1], i_lower[2]},
                                                         {i_upper[0] + 1, i_upper[1] + 1, i_upper[2] + 1}};
    Array<T, NDIMS> res(i_box, true);
    res.reset(std::shared_ptr<T>(p_data.getPointer(depth), simpla::tags::do_nothing()));
    res.DoSetUp();
    return res;
};

template <int NDIMS, typename T>
std::shared_ptr<data::DataBlock> create_simpla_datablock(int IFORM, boost::shared_ptr<SAMRAI::hier::PatchData> pd) {
    std::shared_ptr<data::DataMultiArray<T, NDIMS>> res = nullptr;
    typedef Array<T, NDIMS> array_type;

    switch (IFORM) {
        case VERTEX: {
            auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::NodeData<T>>(pd);
            int depth = p_data->getDepth();
            res = std::make_shared<data::DataMultiArray<T, NDIMS>>(depth);
            for (int d = 0; d < depth; ++d) {
                create_array<T, NDIMS>(p_data->getArrayData(), d).swap(res->GetArray(d));
            }
            break;
        }
        case EDGE: {
            auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::EdgeData<T>>(pd);
            int depth = p_data->getDepth();
            res = std::make_shared<data::DataMultiArray<T, NDIMS>>(depth * 3);
            for (int axis = 0; axis < 3; ++axis) {
                for (int d = 0; d < depth; ++d) {
                    create_array<T, NDIMS>(p_data->getArrayData(axis), d).swap(res->GetArray(axis * depth + d));
                }
            }
            break;
        }
        case FACE: {
            auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::SideData<T>>(pd);
            int depth = p_data->getDepth();
            res = std::make_shared<data::DataMultiArray<T, NDIMS>>(depth * 3);
            for (int axis = 0; axis < 3; ++axis) {
                for (int d = 0; d < depth; ++d) {
                    create_array<T, NDIMS>(p_data->getArrayData(axis), d).swap(res->GetArray(axis * depth + d));
                }
            }
            break;
        }
        case VOLUME: {
            auto p_data = boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<T>>(pd);
            int depth = p_data->getDepth();
            res = std::make_shared<data::DataMultiArray<T, NDIMS>>(depth);
            for (int d = 0; d < depth; ++d) {
                create_array<T, NDIMS>(p_data->getArrayData(), d).swap(res->GetArray(d));
            }
            break;
        }
        default: {
            UNIMPLEMENTED;
            break;
        }
    }
    return res;
}

template <int NDIMS>
std::shared_ptr<data::DataBlock> create_simpla_datablock(const std::shared_ptr<engine::AttributeDesc> &desc,
                                                         boost::shared_ptr<SAMRAI::hier::PatchData> pd) {
    std::shared_ptr<data::DataBlock> res(nullptr);
    if (desc->value_type_info() == (typeid(float))) {
        res = create_simpla_datablock<NDIMS, float>(desc->GetIFORM(), pd);
    } else if (desc->value_type_info() == (typeid(double))) {
        res = create_simpla_datablock<NDIMS, double>(desc->GetIFORM(), pd);
    } else if (desc->value_type_info() == (typeid(int))) {
        res = create_simpla_datablock<NDIMS, int>(desc->GetIFORM(), pd);
    } else {
        RUNTIME_ERROR << "Unsupported m_value_ value_type_info" << std::endl;
    }
    return res;
}
boost::shared_ptr<SAMRAI::hier::PatchData> convert_from_data_block(const std::shared_ptr<engine::AttributeDesc> &desc,
                                                                   std::shared_ptr<data::DataBlock> t) {
    //    UNIMPLEMENTED;
    return nullptr;
}

}  // namespace detail

/** Register conserved variables  and  register plot data with VisIt. */
void SAMRAIHyperbolicPatchStrategyAdapter::registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator) {
    ASSERT(integrator != nullptr);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

    SAMRAI::hier::IntVector d_nghosts{d_dim, 4};
    SAMRAI::hier::IntVector d_fluxghosts{d_dim, 1};
    //**************************************************************
    auto attr_desc = m_ctx_->CollectRegisteredAttributes();
    for (auto const &item : attr_desc) {
        boost::shared_ptr<SAMRAI::hier::Variable> var = simpla::detail::create_samrai_variable(item.second);
        if (var == nullptr) { continue; }

        m_samrai_variables_[item.second] = var;

        /*** NOTE:
        *  1. SAMRAI Visit Writer only support NODE and CELL variable (double,float ,int)
        *  2. SAMRAI   SAMRAI::algs::HyperbolicLevelIntegrator->registerVariable only support double
        **/
        SAMRAI::algs::HyperbolicLevelIntegrator::HYP_VAR_TYPE v_type =
            SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP;
        SAMRAI::hier::IntVector ghosts = d_nghosts;
        std::string coarsen_name = "NO_REFINE";
        std::string refine_name = "NO_REFINE";

        if (item.second->db()->Check("COORDINATES", true) || item.second->db()->Check("INPUT", true)) {
            v_type = SAMRAI::algs::HyperbolicLevelIntegrator::INPUT;
        }
        if (item.second->db()->Check("FLUX", true)) {
            ghosts = d_fluxghosts;
            v_type = SAMRAI::algs::HyperbolicLevelIntegrator::FLUX;
            coarsen_name = "CONSERVATIVE_COARSEN";
            refine_name = "NO_REFINE";
        }
        if (item.second->db()->Check("INPUT", true)) {
            coarsen_name = "NO_REFINE";
            refine_name = "NO_REFINE";
        }
        integrator->registerVariable(var, ghosts, v_type, d_grid_geometry, "", coarsen_name);
        if (item.second->GetPrefix()[0] != '_') {
            std::string visit_variable_type;
            if ((item.second->GetIFORM() == VERTEX || item.second->GetIFORM() == VOLUME) &&
                (item.second->GetDOF() == 1)) {
                visit_variable_type = "SCALAR";
            } else if (((item.second->GetIFORM() == EDGE || item.second->GetIFORM() == FACE) &&
                        (item.second->GetDOF() == 1)) ||
                       ((item.second->GetIFORM() == VERTEX || item.second->GetIFORM() == VOLUME) &&
                        (item.second->GetDOF() == 3))) {
                visit_variable_type = "VECTOR";
            } else if (((item.second->GetIFORM() == VERTEX || item.second->GetIFORM() == VOLUME) &&
                        item.second->GetDOF() == 9) ||
                       ((item.second->GetIFORM() == EDGE || item.second->GetIFORM() == FACE) &&
                        item.second->GetDOF() == 3)) {
                visit_variable_type = "TENSOR";
            } else {
                WARNING << "Can not register attribute [" << item.second->GetPrefix() << "] to VisIt writer !"
                        << std::endl;
            }

            if (visit_variable_type != "" && item.second->db()->Check("COORDINATES", true)) {
                d_visit_writer->registerNodeCoordinates(
                    vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
            } else if (item.second->GetIFORM() == VERTEX || item.second->GetIFORM() == VOLUME) {
                d_visit_writer->registerPlotQuantity(
                    item.second->GetPrefix(), visit_variable_type,
                    vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
            }
        }
    }
    //    integrator->printClassData(std::cout);
    //    vardb->printClassData(std::cout);
}
void SAMRAIHyperbolicPatchStrategyAdapter::ConvertPatchFromSAMRAI(SAMRAI::hier::Patch &patch, engine::Patch *p) {
    p->SetBlock(std::make_shared<engine::MeshBlock>(
        index_box_type{{patch.getBox().lower()[0], patch.getBox().lower()[1], patch.getBox().lower()[2]},
                       {patch.getBox().upper()[0] + 1, patch.getBox().upper()[1] + 1, patch.getBox().upper()[2] + 1}},
        patch.getPatchLevelNumber()));

    for (auto &item : m_samrai_variables_) {
        auto samrai_id =
            SAMRAI::hier::VariableDatabase::getDatabase()->mapVariableAndContextToIndex(item.second, getDataContext());

        if (!patch.checkAllocated(samrai_id)) { patch.allocatePatchData(samrai_id); }

        p->Push(item.first->GetID(),
                simpla::detail::create_simpla_datablock<NDIMS>(item.first, patch.getPatchData(samrai_id)));
    }
}
void SAMRAIHyperbolicPatchStrategyAdapter::ConvertPatchToSAMRAI(SAMRAI::hier::Patch &patch, engine::Patch *p) {
    //    for (auto &item : m_samrai_variables_) {
    //        auto samrai_id =
    //            SAMRAI::hier::VariableDatabase::getDatabase()->mapVariableAndContextToIndex(item.second,
    //            getDataContext());
    //        if (!patch.checkAllocated(samrai_id)) { patch.allocatePatchData(samrai_id); }
    //        patch.setPatchData(samrai_id,
    //                           simpla::detail::convert_from_data_block(*item.first,
    //                           p->ConvertPatchToSAMRAI(item.first->GetGUID())));
    //    }
}
/** Set up parameters for nonuniform load balancing, if used. */
void SAMRAIHyperbolicPatchStrategyAdapter::setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                                                             SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm) {
    const SAMRAI::hier::IntVector &zero_vec = SAMRAI::hier::IntVector::getZero(d_dim);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();
    if (d_use_nonuniform_workload && (gridding_algorithm != nullptr)) {
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
 * Set initial data for solution variables on patch interior.
 * This routine is called whenever a new patch is introduced to the
 * AMR patch hierarchy.  Note that the routine does nothing unless
 * we are at the initial time.  In all other cases, conservative
 * interpolation from coarser levels and copies from patches at the
 * same mesh resolution are sufficient to set data.
 *
 */
void SAMRAIHyperbolicPatchStrategyAdapter::initializeDataOnPatch(SAMRAI::hier::Patch &patch, double data_time,
                                                                 bool initial_time) {
    if (initial_time) {
        auto p = m_ctx_->GetAtlas().Pop(static_cast<id_type>(patch.getLocalId().getValue()));
        ConvertPatchFromSAMRAI(patch, p.get());

        index_tuple gw = p->GetBlock()->GetGhostWidth();

        auto pgeom = boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());

        for (int n = 1; n <= 3; ++n)
            for (auto const &b : pgeom->getCodimensionBoundaries(n)) {
                index_box_type box{{b.getBox().lower()[0], b.getBox().lower()[1], b.getBox().lower()[2]},
                                   {b.getBox().upper()[0], b.getBox().upper()[1], b.getBox().upper()[2]}};

                index_box_type vertex_box = box;
                index_box_type volume_box = box;
                index_box_type edge0_box = box;
                index_box_type edge1_box = box;
                index_box_type edge2_box = box;
                index_box_type face0_box = box;
                index_box_type face1_box = box;
                index_box_type face2_box = box;

                for (int j = 0; j < 3; ++j) {
                    switch (b.getBoundaryOrientation(j)) {
                        case SAMRAI::hier::BoundaryBox::LOWER:
                            std::get<0>(vertex_box)[j] -= gw[j];
                            std::get<1>(vertex_box)[j] += 1;
                            std::get<0>(volume_box)[j] -= gw[j];
                            std::get<1>(volume_box)[j] += 1;

                            std::get<0>(edge0_box)[j] -= gw[j];
                            std::get<0>(edge1_box)[j] -= gw[j];
                            std::get<0>(edge2_box)[j] -= gw[j];

                            std::get<1>(edge0_box)[j] += 1;
                            std::get<1>(edge1_box)[j] += 1;
                            std::get<1>(edge2_box)[j] += 1;

                            std::get<0>(face0_box)[j] -= gw[j];
                            std::get<0>(face1_box)[j] -= gw[j];
                            std::get<0>(face2_box)[j] -= gw[j];

                            std::get<1>(face0_box)[j] += 1;
                            std::get<1>(face1_box)[j] += 1;
                            std::get<1>(face2_box)[j] += 1;

                            break;

                        case SAMRAI::hier::BoundaryBox::UPPER:

                            std::get<0>(vertex_box)[j] += 1;
                            std::get<1>(vertex_box)[j] += gw[j] + 1;
                            std::get<1>(volume_box)[j] += gw[j];

                            std::get<0>(edge0_box)[j] += (j == 0) ? 0 : 1;
                            std::get<0>(edge1_box)[j] += (j == 1) ? 0 : 1;
                            std::get<0>(edge2_box)[j] += (j == 2) ? 0 : 1;

                            std::get<1>(edge0_box)[j] += gw[j] + ((j == 0) ? 1 : 2);
                            std::get<1>(edge1_box)[j] += gw[j] + ((j == 1) ? 1 : 2);
                            std::get<1>(edge2_box)[j] += gw[j] + ((j == 2) ? 1 : 2);

                            std::get<0>(face0_box)[j] += (j == 0) ? 1 : 0;
                            std::get<0>(face1_box)[j] += (j == 1) ? 1 : 0;
                            std::get<0>(face2_box)[j] += (j == 2) ? 1 : 0;

                            std::get<1>(face0_box)[j] += gw[j] + ((j == 0) ? 2 : 1);
                            std::get<1>(face1_box)[j] += gw[j] + ((j == 1) ? 2 : 1);
                            std::get<1>(face2_box)[j] += gw[j] + ((j == 2) ? 2 : 1);

                            break;
                        case SAMRAI::hier::BoundaryBox::MIDDLE:
                            std::get<1>(vertex_box)[j] += 1;
                            std::get<1>(volume_box)[j] += 1;

                            std::get<1>(edge0_box)[j] += (j == 0) ? 1 : 2;
                            std::get<1>(edge1_box)[j] += (j == 1) ? 1 : 2;
                            std::get<1>(edge2_box)[j] += (j == 2) ? 1 : 2;

                            std::get<1>(face0_box)[j] += (j == 0) ? 2 : 1;
                            std::get<1>(face1_box)[j] += (j == 1) ? 2 : 1;
                            std::get<1>(face2_box)[j] += (j == 2) ? 2 : 1;

                        default:
                            break;
                    }
                }
                //                CHECK(p->GetBlock()->GetIndexBox());
                //                CHECK(vertex_box);
                //                CHECK(edge0_box);
                //                CHECK(edge1_box);
                //                CHECK(edge2_box);
                //                CHECK(face0_box);
                //                CHECK(face1_box);
                //                CHECK(face2_box);
                //                CHECK(volume_box);

                p->m_ranges_[std::string(EntityIFORMName[VERTEX]) + "_PATCH_BOUNDARY"].append(
                    std::make_shared<ContinueRange<EntityId>>(vertex_box, 0));

                p->m_ranges_[std::string(EntityIFORMName[EDGE]) + "_PATCH_BOUNDARY"]
                    .append(std::make_shared<ContinueRange<EntityId>>(edge0_box, 1))
                    .append(std::make_shared<ContinueRange<EntityId>>(edge1_box, 2))
                    .append(std::make_shared<ContinueRange<EntityId>>(edge2_box, 4));

                p->m_ranges_[std::string(EntityIFORMName[FACE]) + "_PATCH_BOUNDARY"]
                    .append(std::make_shared<ContinueRange<EntityId>>(face0_box, 6))
                    .append(std::make_shared<ContinueRange<EntityId>>(face1_box, 5))
                    .append(std::make_shared<ContinueRange<EntityId>>(face2_box, 3));

                p->m_ranges_[std::string(EntityIFORMName[VOLUME]) + "_PATCH_BOUNDARY"].append(
                    std::make_shared<ContinueRange<EntityId>>(volume_box, 7));
            }
        m_ctx_->InitialCondition(p.get(), data_time);

        //        m_ctx_->GetMesh()->Push(p.get());
        //        VERBOSE << "Initialize Mesh : " << m_ctx_->GetMesh()->GetRegisterName() << std::endl;
        //        m_ctx_->GetMesh()->InitializeData(data_time);
        //        for (auto const &item : m_ctx_->GetModel().GetAll()) {
        //            m_ctx_->GetMesh()->RegisterRanges(item.second, item.first);
        //        }
        //
        //        for (auto &d : m_ctx_->GetAllDomains()) {
        //            VERBOSE << "Initialize Domain : " << d.first << std::endl;
        //            d.second->DoInitialCondition(p.get(), data_time);
        //        }
        //        m_ctx_->GetMesh()->Pop(p.get());
    }

    if (d_use_nonuniform_workload) {
        if (!patch.checkAllocated(d_workload_data_id)) { patch.allocatePatchData(d_workload_data_id); }
        boost::shared_ptr<SAMRAI::pdat::CellData<double>> workload_data(
            boost::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
                patch.getPatchData(d_workload_data_id)));
        TBOX_ASSERT(workload_data);

        const SAMRAI::hier::Box &box = patch.getBox();
        const SAMRAI::hier::BoxId &box_id = box.getBoxId();
        const SAMRAI::hier::LocalId &local_id = box_id.getLocalId();
        double id_val = (local_id.getValue() % 2 == 0) ? static_cast<double>(local_id.getValue() % 10) : 0.0;
        workload_data->fillAll(1.0 + id_val);
    }
}

/**************************************************************************
 *
 * Compute stable time increment for patch.  Return this m_value_.
 *
 *************************************************************************
 */

double SAMRAIHyperbolicPatchStrategyAdapter::computeStableDtOnPatch(SAMRAI::hier::Patch &patch, bool initial_time,
                                                                    double dt_time) {
    //    auto pgeom =
    //    boost::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
    //    return pgeom->getDx()[0] / 2.0;
    return dt_time;
}

/**************************************************************************
 *
 * Compute time integral of numerical fluxes for finite difference
 * at each cell face on patch.  When d_dim == tbox::Dimension(3)), there are two options
 * for the transverse flux correction.  Otherwise, there is only one.
 *
 *************************************************************************
 */

void SAMRAIHyperbolicPatchStrategyAdapter::computeFluxesOnPatch(SAMRAI::hier::Patch &patch, double time, double dt) {}

/**************************************************************************
 *
 * Update solution variables by performing a conservative
 * difference with the fluxes calculated in computeFluxesOnPatch().
 *
 **************************************************************************/

void SAMRAIHyperbolicPatchStrategyAdapter::conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, double time_now,
                                                                         double time_dt, bool at_syncronization) {
    auto p = m_ctx_->GetAtlas().Pop(static_cast<id_type>(patch.getLocalId().getValue()));

    ConvertPatchFromSAMRAI(patch, p.get());
    m_ctx_->GetMesh()->Push(p.get());
    m_ctx_->GetMesh()->SetBoundaryCondition(time_now, time_dt);
    m_ctx_->Advance(p.get(), time_now, time_dt);
    //    m_ctx_->GetMesh()->Push(p.get());
    //    for (auto &d : m_ctx_->GetAllDomains()) { d.second->DoAdvance(p.get(), time_now, time_dt); }
    m_ctx_->GetMesh()->Pop(p.get());
}

/**************************************************************************
 *
 * Tag cells for refinement using Richardson extrapolation.  Criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAIHyperbolicPatchStrategyAdapter::tagRichardsonExtrapolationCells(
    SAMRAI::hier::Patch &patch, const int error_level_number,
    const boost::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
    const boost::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse, double regrid_time, double deltat,
    const int error_coarsen_ratio, bool initial_error, const int tag_index, bool uses_gradient_detector_too) {}

/**************************************************************************
 *
 * Tag cells for refinement using gradient detector.  Tagging criteria
 * defined in input.
 *
 *************************************************************************
 */

void SAMRAIHyperbolicPatchStrategyAdapter::tagGradientDetectorCells(SAMRAI::hier::Patch &patch, double regrid_time,
                                                                    bool initial_error, int tag_indx,
                                                                    bool uses_richardson_extrapolation_too) {}

void SAMRAIHyperbolicPatchStrategyAdapter::setPhysicalBoundaryConditions(
    SAMRAI::hier::Patch &patch, double fill_time, const SAMRAI::hier::IntVector &ghost_width_to_fill) {
    auto p = m_ctx_->GetAtlas().Pop(static_cast<id_type>(patch.getLocalId().getValue()));
    ConvertPatchFromSAMRAI(patch, p.get());
    m_ctx_->GetMesh()->Push(p.get());
    m_ctx_->GetMesh()->SetBoundaryCondition(fill_time, 0);
    for (auto &d : m_ctx_->GetAllDomains()) { d.second->DoBoundaryCondition(p.get(), fill_time, 0); }
    m_ctx_->GetMesh()->Pop(p.get());
}

/**************************************************************************
 *
 * Register VisIt SetDataBlock writer to write GetDataBlock to plot files that may
 * be postprocessed by the VisIt tool.
 *
 **************************************************************************/

void SAMRAIHyperbolicPatchStrategyAdapter::registerVisItDataWriter(
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer) {
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}

/**************************************************************************
 * Write SAMRAIWorkerAdapter object state to specified stream.
 **************************************************************************/

void SAMRAIHyperbolicPatchStrategyAdapter::printClassData(std::ostream &os) const {
    os << "\nSAMRAIWorkerAdapter::printClassData..." << std::endl;
    os << "m_domain_geo_prefix_ = " << m_name_ << std::endl;
    os << "d_grid_geometry = " << d_grid_geometry.get() << std::endl;
    os << std::endl;
}
struct SAMRAITimeIntegrator::pimpl_s {
    boost::shared_ptr<SAMRAIHyperbolicPatchStrategyAdapter> hyperbolic_patch_strategy;
    boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;
    boost::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;
    boost::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

    //    boost::shared_ptr<SAMRAI::engine::StandardTagAndInitialize> error_detector;
    //    boost::shared_ptr<SAMRAI::engine::BergerRigoutsos> box_generator;
    //    boost::shared_ptr<SAMRAI::engine::CascadePartitioner> load_balancer;
    //    boost::shared_ptr<SAMRAI::engine::GriddingAlgorithm> gridding_algorithm;
    boost::shared_ptr<SAMRAI::algs::TimeRefinementIntegrator> m_time_refinement_integrator_;
    // VisItDataWriter is only present if HDF is available
    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> visit_data_writer_;

    bool write_restart = false;
    int restart_interval = 0;

    std::string restart_write_dirname;

    bool viz_dump_data = false;
    int viz_dump_interval = 1;

    unsigned int ndims = 3;
};
SAMRAITimeIntegrator::SAMRAITimeIntegrator() : m_pimpl_(new pimpl_s) {}
SAMRAITimeIntegrator::~SAMRAITimeIntegrator() {
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
}
void SAMRAITimeIntegrator::Initialize() {
    dcomplex a = std::numeric_limits<dcomplex>::signaling_NaN();
    engine::TimeIntegrator::Initialize();
    /** Setup SAMRAI::tbox::MPI.      */
    SAMRAI::tbox::SAMRAI_MPI::init(*reinterpret_cast<MPI_Comm const *>(GLOBAL_COMM.comm()));  //
    SAMRAI::tbox::SAMRAIManager::initialize();
    /** Setup SAMRAI, enable logging, and process command line.     */
    SAMRAI::tbox::SAMRAIManager::startup();

    //    data::DataTable(std::make_shared<DataBackendSAMRAI>()).swap(*db());
    //    const SAMRAI::tbox::SAMRAI_MPI & mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());
}
void SAMRAITimeIntegrator::Synchronize() { engine::TimeIntegrator::Synchronize(); }
std::shared_ptr<data::DataTable> SAMRAITimeIntegrator::Serialize() const { return engine::TimeIntegrator::Serialize(); }
void SAMRAITimeIntegrator::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    engine::TimeIntegrator::Deserialize(cfg);
}
void SAMRAITimeIntegrator::TearDown() { engine::TimeIntegrator::TearDown(); }
void SAMRAITimeIntegrator::Update() {
    engine::TimeIntegrator::Update();
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

    auto &ctx = GetContext();
    auto &atlas = ctx->GetAtlas();
    auto p_mesh = ctx->GetMesh();
    unsigned int ndims = static_cast<unsigned int>(ctx->GetMesh()->GetNDims());
    bool use_refined_timestepping = true;  // m_samrai_db_->GetValue<bool>("use_refined_timestepping", true);

    SAMRAI::tbox::Dimension dim(static_cast<unsigned short>(ndims));

    //    samrai_db = simpla::detail::convert_database(db(), name());
    /**
     * Create major algorithm and data objects which comprise application.
     * Each object will be initialized either from input data or restart
     * files, or a combination of both.  Refer to each class constructor
     * for details.  For more information on the composition of objects
     * for this application, see comments at top of file.
     */

    auto CartesianGridGeometry = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("CartesianGeometry");

    nTuple<int, 3> i_low{0, 0, 0};
    nTuple<int, 3> i_up{0, 0, 0};

    i_low = p_mesh->GetIndexOffset();
    i_up = i_low + p_mesh->GetDimensions();

    SAMRAI::tbox::DatabaseBox box{SAMRAI::tbox::Dimension(3), &i_low[0], &i_up[0]};
    CartesianGridGeometry->putDatabaseBox("domain_boxes_0", box);
    nTuple<int, 3> periodic_dimension{0, 0, 0};
    periodic_dimension = p_mesh->GetPeriodicDimension();
    nTuple<double, 3> x_low = p_mesh->point(
        EntityId{static_cast<int16_t>(i_low[0]), static_cast<int16_t>(i_low[1]), static_cast<int16_t>(i_low[2]), 0});
    nTuple<double, 3> x_up = p_mesh->point(
        EntityId{static_cast<int16_t>(i_up[0]), static_cast<int16_t>(i_up[1]), static_cast<int16_t>(i_up[2]), 0});

    CartesianGridGeometry->putIntegerArray("periodic_dimension", &periodic_dimension[0], ndims);
    CartesianGridGeometry->putDoubleArray("x_lo", &x_low[0], ndims);
    CartesianGridGeometry->putDoubleArray("x_up", &x_up[0], ndims);

    m_pimpl_->grid_geometry.reset(
        new SAMRAI::geom::CartesianGridGeometry(dim, "CartesianGeometry", CartesianGridGeometry));

    //---------------------------------

    auto PatchHierarchy = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("PatchHierarchy");

    // Maximum number of levels in hierarchy.
    PatchHierarchy->putInteger("max_levels", static_cast<int>(atlas.GetMaxLevel()));

    auto ratio_to_coarser = PatchHierarchy->putDatabase("ratio_to_coarser");

    for (int i = 0, n = static_cast<int>(atlas.GetMaxLevel()); i < n; ++i) {
        nTuple<int, 3> level;
        level = atlas.GetRefineRatio(i);
        ratio_to_coarser->putIntegerArray("level_" + std::to_string(i), &level[0], ndims);
    }

    auto largest_patch_size = PatchHierarchy->putDatabase("largest_patch_size");
    auto smallest_patch_size = PatchHierarchy->putDatabase("smallest_patch_size");

    nTuple<int, 3> level_largest, level_smallest;
    level_smallest = atlas.GetSmallestDimensions();
    level_largest = atlas.GetLargestDimensions();

    smallest_patch_size->putIntegerArray("level_0", &level_smallest[0], ndims);
    largest_patch_size->putIntegerArray("level_0", &level_largest[0], ndims);

    m_pimpl_->patch_hierarchy.reset(
        new SAMRAI::hier::PatchHierarchy("PatchHierarchy", m_pimpl_->grid_geometry, PatchHierarchy));

    auto HyperbolicLevelIntegrator = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("HyperbolicLevelIntegrator");

    // Refer to algs::HyperbolicLevelIntegrator for input
    // max cfl factor used in problem
    HyperbolicLevelIntegrator->putDouble("cfl", engine::TimeIntegrator::GetCFL());
    HyperbolicLevelIntegrator->putDouble("cfl_init", 0.9);
    HyperbolicLevelIntegrator->putBool("lag_dt_computation", true);
    HyperbolicLevelIntegrator->putBool("use_ghosts_to_compute_dt", true);

    /**
     *  create m_pimpl_->hyp_level_integrator and error_detector
     */
    m_pimpl_->hyperbolic_patch_strategy.reset(
        new SAMRAIHyperbolicPatchStrategyAdapter(GetContext(), m_pimpl_->grid_geometry));

    m_pimpl_->hyp_level_integrator.reset(new SAMRAI::algs::HyperbolicLevelIntegrator(
        "SAMRAILevelIntegrator", HyperbolicLevelIntegrator, m_pimpl_->hyperbolic_patch_strategy.get(),
        use_refined_timestepping));

    auto StandardTagAndInitialize = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("StandardTagAndInitialize");
    // Refer to mesh::StandardTagAndInitialize for input
    StandardTagAndInitialize->putString("tagging_method", "GRADIENT_DETECTOR");

    auto error_detector = boost::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
        "StandardTagAndInitialize", m_pimpl_->hyp_level_integrator.get(), StandardTagAndInitialize);
    /**
     *  create grid_algorithm
     */
    auto BergerRigoutsos = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("BergerRigoutsos");

    BergerRigoutsos->putBool("sort_output_nodes", true);       // Makes results repeatable.
    BergerRigoutsos->putDouble("efficiency_tolerance", 0.85);  // min % of GetTag cells in new patch level,
    BergerRigoutsos->putDouble("combine_efficiency", 0.95);    //  chop box if  sum of volumes of   smaller
    // boxes <  efficiency * vol of large box

    auto box_generator = boost::make_shared<SAMRAI::mesh::BergerRigoutsos>(dim, BergerRigoutsos);

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto LoadBalancer = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("LoadBalancer");
    auto load_balancer = boost::make_shared<SAMRAI::mesh::CascadePartitioner>(dim, "LoadBalancer", LoadBalancer);

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());
    //    load_balancer->printStatistics(std::cout);

    auto GriddingAlgorithm = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("GriddingAlgorithm");

    auto gridding_algorithm = boost::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
        m_pimpl_->patch_hierarchy, "GriddingAlgorithm", GriddingAlgorithm, error_detector, box_generator,
        load_balancer);

    // Refer to algs::TimeRefinementIntegrator for input
    auto TimeRefinementIntegrator = boost::make_shared<SAMRAI::tbox::MemoryDatabase>("TimeRefinementIntegrator");

    TimeRefinementIntegrator->putDouble("start_time", engine::TimeIntegrator::GetTimeNow());  // initial simulation time
    TimeRefinementIntegrator->putDouble("end_time", engine::TimeIntegrator::GetTimeEnd());    // final simulation time
    TimeRefinementIntegrator->putDouble("grow_dt", 1.1);  // growth factor for timesteps
    TimeRefinementIntegrator->putInteger("max_integrator_steps", 100);

    m_pimpl_->m_time_refinement_integrator_.reset(new SAMRAI::algs::TimeRefinementIntegrator(
        "TimeRefinementIntegrator", TimeRefinementIntegrator, m_pimpl_->patch_hierarchy, m_pimpl_->hyp_level_integrator,
        gridding_algorithm));

    m_pimpl_->visit_data_writer_.reset(
        new SAMRAI::appu::VisItDataWriter(dim, "SimPLA VisIt Writer", GetOutputURL(), 1));

    m_pimpl_->hyperbolic_patch_strategy->registerVisItDataWriter(m_pimpl_->visit_data_writer_);

    m_pimpl_->m_time_refinement_integrator_->initializeHierarchy();

    m_pimpl_->grid_geometry->printClassData(std::cout);
    m_pimpl_->hyp_level_integrator->printClassData(std::cout);
    //    m_pimpl_->m_time_refinement_integrator_->printClassData(std::cout);
    MESSAGE << "==================  Context is initialized!  =================" << std::endl;
};
void SAMRAITimeIntegrator::Finalize() {
    m_pimpl_->visit_data_writer_.reset();
    m_pimpl_->m_time_refinement_integrator_.reset();
    m_pimpl_->hyp_level_integrator.reset();
    m_pimpl_->hyperbolic_patch_strategy.reset();
    engine::TimeIntegrator::Finalize();
}
Real SAMRAITimeIntegrator::Advance(Real time_dt) {
    ASSERT(m_pimpl_->m_time_refinement_integrator_ != nullptr);

    // SetTimeNow(m_pimpl_->m_time_refinement_integrator->getIntegratorTime());
    Real loop_time = GetTimeNow();
    Real loop_time_end = std::min(loop_time + time_dt, GetTimeEnd());
    Real loop_dt = time_dt;
    while ((loop_time < loop_time_end) &&
           (loop_dt > 0)) {  //&& m_pimpl_->m_time_refinement_integrator->stepsRemaining() > 0
        Real dt_new = m_pimpl_->m_time_refinement_integrator_->advanceHierarchy(loop_dt, false);
        loop_dt = std::min(dt_new, loop_time_end - loop_time);
        loop_time += loop_dt;
    }

    SetTimeNow(loop_time_end);
    return loop_time_end;
}
void SAMRAITimeIntegrator::CheckPoint() const {
    if (m_pimpl_->visit_data_writer_ != nullptr) {
        // VERBOSE << "Check Point at Step " << m_pimpl_->m_time_refinement_integrator->getIntegratorStep() <<
        // std::endl;
        m_pimpl_->visit_data_writer_->writePlotData(m_pimpl_->patch_hierarchy,
                                                    m_pimpl_->m_time_refinement_integrator_->getIntegratorStep(),
                                                    m_pimpl_->m_time_refinement_integrator_->getIntegratorTime());
    }
}
void SAMRAITimeIntegrator::Dump() const {
    //    if (m_pimpl_->visit_data_writer != nullptr) {
    //        VERBOSE << "Dump : Step = " << GetNumberOfStep() << std::end;
    //        m_pimpl_->visit_data_writer->writePlotData(m_pimpl_->patch_hierarchy,
    //        m_pimpl_->m_time_refinement_integrator->getIntegratorStep(),
    //                                         m_pimpl_->m_time_refinement_integrator->getIntegratorTime());
    //    }
}
bool SAMRAITimeIntegrator::Done() const {
    // m_pimpl_->m_time_refinement_integrator != nullptr ? !m_pimpl_->m_time_refinement_integrator->stepsRemaining():;
    return engine::TimeIntegrator::Done();
}
}  // namespace simpla