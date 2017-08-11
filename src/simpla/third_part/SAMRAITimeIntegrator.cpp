//
// Created by salmon on 16-10-24.
//

#include "simpla/SIMPLA_config.h"

#include "SAMRAITimeIntegrator.h"
// Headers for SimPla
#include <cmath>
#include <map>
#include <memory>
#include <string>

#include "simpla/algebra/Algebra.h"
#include "simpla/data/Data.h"
#include "simpla/engine/Engine.h"
#include "simpla/engine/Mesh.h"
#include "simpla/parallel/MPIComm.h"
#include "simpla/particle/ParticleDataBlock.h"
#include "simpla/utilities/Log.h"

// Headers for SAMRAI
#include "SAMRAI/SAMRAI_config.h"

#include "SAMRAI/algs/HyperbolicLevelIntegrator.h"
#include "SAMRAI/algs/TimeRefinementIntegrator.h"
#include "SAMRAI/algs/TimeRefinementLevelStrategy.h"

#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"

#include "SAMRAI/hier/BoundaryBox.h"
#include "SAMRAI/hier/BoxContainer.h"
#include "SAMRAI/hier/Index.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/hier/MultiblockBoxTree.h"
#include "SAMRAI/hier/PatchData.h"
#include "SAMRAI/hier/PatchDataFactory.h"
#include "SAMRAI/hier/PatchDataRestartManager.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/Variable.h"
#include "SAMRAI/hier/VariableDatabase.h"

#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include "SAMRAI/pdat/CellData.h"
#include "SAMRAI/pdat/CellIndex.h"
#include "SAMRAI/pdat/CellIterator.h"
#include "SAMRAI/pdat/CellVariable.h"
#include "SAMRAI/pdat/EdgeVariable.h"
#include "SAMRAI/pdat/FaceData.h"
#include "SAMRAI/pdat/FaceIndex.h"
#include "SAMRAI/pdat/FaceVariable.h"
#include "SAMRAI/pdat/NodeVariable.h"
#include "SAMRAI/pdat/SparseDataVariable.h"

#include "SAMRAI/tbox/BalancedDepthFirstTree.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/Dimension.h"
#include "SAMRAI/tbox/InputDatabase.h"
#include "SAMRAI/tbox/InputManager.h"
#include "SAMRAI/tbox/MathUtilities.h"
#include "SAMRAI/tbox/PIO.h"
#include "SAMRAI/tbox/RestartManager.h"
#include "SAMRAI/tbox/SAMRAIManager.h"
#include "SAMRAI/tbox/SAMRAI_MPI.h"
#include "SAMRAI/tbox/Utilities.h"

#include "SAMRAI/appu/BoundaryUtilityStrategy.h"
#include "SAMRAI/appu/CartesianBoundaryDefines.h"
#include "SAMRAI/appu/CartesianBoundaryUtilities2.h"
#include "SAMRAI/appu/CartesianBoundaryUtilities3.h"
#include "SAMRAI/appu/VisItDataWriter.h"
#include "SAMRAI/pdat/NodeDoubleLinearTimeInterpolateOp.h"
#include "SAMRAI/pdat/SideVariable.h"

namespace simpla {

struct spParticlePatchData : public SAMRAI::hier::PatchData {
   public:
    spParticlePatchData(const SAMRAI::hier::Box &box, int depth, const SAMRAI::hier::IntVector &ghosts)
        : SAMRAI::hier::PatchData(box, ghosts), m_depth_(depth), m_data_block_(new ParticleDataBlock(depth)) {}
    ~spParticlePatchData() override {}

    virtual void copy(const SAMRAI::hier::PatchData &src) override {
        auto const &p = dynamic_cast<spParticlePatchData const &>(src);
        m_data_block_ = p.m_data_block_;
    };

    virtual void copy2(SAMRAI::hier::PatchData &dst) const override {
        auto &p = dynamic_cast<spParticlePatchData &>(dst);
        p.m_data_block_ = m_data_block_;
    };

    virtual void copy(const SAMRAI::hier::PatchData &src, const SAMRAI::hier::BoxOverlap &overlap) override{
        //        UNIMPLEMENTED;
    };

    virtual void copy2(SAMRAI::hier::PatchData &dst, const SAMRAI::hier::BoxOverlap &overlap) const override{
        //        UNIMPLEMENTED;
    };

    virtual bool canEstimateStreamSizeFromBox() const override { return false; };

    virtual size_t getDataStreamSize(const SAMRAI::hier::BoxOverlap &overlap) const override { return 0; };

    virtual void packStream(SAMRAI::tbox::MessageStream &stream,
                            const SAMRAI::hier::BoxOverlap &overlap) const override {
        UNIMPLEMENTED;
    };

    virtual void unpackStream(SAMRAI::tbox::MessageStream &stream, const SAMRAI::hier::BoxOverlap &overlap) override {
        UNIMPLEMENTED;
    };

    virtual void getFromRestart(const std::shared_ptr<SAMRAI::tbox::Database> &restart_db) override { UNIMPLEMENTED; }

    virtual void putToRestart(const std::shared_ptr<SAMRAI::tbox::Database> &restart_db) const override {
        UNIMPLEMENTED;
    }
    int getDepth() const { return m_depth_; }
    std::shared_ptr<data::DataBlock> GetDataBlock() const { return m_data_block_; }
    void SetDataBlock(std::shared_ptr<data::DataBlock> const &blk) { m_data_block_ = blk; }

   private:
    int m_depth_ = 0;
    std::shared_ptr<data::DataBlock> m_data_block_;
};
/**
 * Class NodeFloatInjection implements constant
 * averaging (i.e., injection) for node-centered float patch data defined
 * over a  mesh.  It is derived from the hier::CoarsenOperator base
 * class.  The numerical operations for theaveraging use FORTRAN numerical
 * routines.
 *
 * @see hier::CoarsenOperator
 */

class spParticleCoarsenRefine : public SAMRAI::hier::CoarsenOperator {
   public:
    /**
     * Uninteresting default constructor.
     */
    spParticleCoarsenRefine() : SAMRAI::hier::CoarsenOperator("CONSTANT_COARSEN"){};

    /**
     * Uninteresting virtual destructor.
     */
    ~spParticleCoarsenRefine() override = default;

    /**
     * The priority of node-centered constant averaging is 0.
     * It will be performed before any user-defined coarsen operations.
     */
    int getOperatorPriority() const override { return 0; }

    /*{
     * The stencil width of the constant averaging operator is the vector of
     * zeros.  That is, its stencil does not extend outside the fine box.
     */
    SAMRAI::hier::IntVector getStencilWidth(const SAMRAI::tbox::Dimension &dim) const override {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    /**
     * Coarsen the source component on the fine patch to the destination
     * component on the coarse patch using the node-centered float constant
     * averaging operator.  Coarsening is performed on the intersection of
     * the destination patch and the coarse box.  It is assumed that the
     * fine patch contains sufficient data for the stencil width of the
     * coarsening operator.
     */
    void coarsen(SAMRAI::hier::Patch &coarse, const SAMRAI::hier::Patch &fine, const int dst_component,
                 const int src_component, const SAMRAI::hier::Box &coarse_box,
                 const SAMRAI::hier::IntVector &ratio) const override {
        UNIMPLEMENTED;
    };
};

/**
* Class ParticleConstantRefine implements constant
* interpolation for cell-centered float patch data defined over a
* mesh.  It is derived from the hier::RefineOperator base class.
* The numerical operations for interpolation use FORTRAN numerical routines.
*
* @see hier::RefineOperator
*/

class spParticleConstantRefine : public SAMRAI::hier::RefineOperator {
   public:
    /**
     * Uninteresting default constructor.
     */
    spParticleConstantRefine() : SAMRAI::hier::RefineOperator("CONSTANT_REFINE") {}

    /**
     * Uninteresting virtual destructor.
     */
    virtual ~spParticleConstantRefine() {}

    /**
     * The priority of cell-centered float constant interpolation is 0.
     * It will be performed before any user-defined interpolation operations.
     */
    int getOperatorPriority() const { return 0; }

    /**
     * The stencil width of the constant interpolation operator is the vector
     * of zeros.  That is, its stencil does not extend outside the fine box.
     */
    SAMRAI::hier::IntVector getStencilWidth(const SAMRAI::tbox::Dimension &dim) const {
        return SAMRAI::hier::IntVector::getZero(dim);
    }

    /**
     * Refine the source component on the coarse patch to the destination
     * component on the fine patch using the cell-centered float constant
     * interpolation operator.  Interpolation is performed on the intersection
     * of the destination patch and the boxes contained in fine_overlap.
     * It is assumed that the coarse patch contains sufficient data for the
     * stencil width of the refinement operator.
     *
     * @pre dynamic_cast<const CellOverlap *>(&fine_overlap) != 0
     */
    void refine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse, const int dst_component,
                const int src_component, const SAMRAI::hier::BoxOverlap &fine_overlap,
                const SAMRAI::hier::IntVector &ratio) const {
        UNIMPLEMENTED;
    }

    /**
     * Refine the source component on the coarse patch to the destination
     * component on the fine patch using the cell-centered float constant
     * interpolation operator.  Interpolation is performed on the intersection
     * of the destination patch and the fine box.   It is assumed that the
     * coarse patch contains sufficient data for the stencil width of the
     * refinement operator.  This differs from the above refine() method
     * only in that it operates on a single fine box instead of a BoxOverlap.
     */
    void refine(SAMRAI::hier::Patch &fine, const SAMRAI::hier::Patch &coarse, const int dst_component,
                const int src_component, const SAMRAI::hier::Box &fine_box,
                const SAMRAI::hier::IntVector &ratio) const {
        UNIMPLEMENTED;
    }
};

/**
 * Class CellDoubleLinearTimeInterpolateOp implements standard
 * linear time interpolation for cell-centered double patch data.
 * It is derived from the hier::TimeInterpolateOperator base class.
 * The interpolation uses FORTRAN numerical routines.
 *
 * @see hier::TimeInterpolateOperator
 */

class spParticleLinearTimeInterpolateOp : public SAMRAI::hier::TimeInterpolateOperator {
   public:
    /**
     * Uninteresting default constructor.
     */
    spParticleLinearTimeInterpolateOp() : SAMRAI::hier::TimeInterpolateOperator() {}

    /**
     * Uninteresting virtual destructor.
     */
    virtual ~spParticleLinearTimeInterpolateOp() = default;

    /**
     * Perform linear time interpolation between two cell-centered double
     * patch data sources and place result in the destination patch data.
     * Time interpolation is performed on the intersection of the destination
     * patch data and the input box.  The time to which data is interpolated
     * is provided by the destination data.
     *
     * @pre dynamic_cast<const CellData<float> *>(&src_data_old) != 0
     * @pre dynamic_cast<const CellData<float> *>(&src_data_new) != 0
     * @pre dynamic_cast<CellData<float> *>(&dst_data) != 0
     * @pre (where * src_data_old->getGhostBox()).isSpatiallyEqual(where)
     * @pre (where * src_data_new->getGhostBox()).isSpatiallyEqual(where)
     * @pre (where * dst_data->getGhostBox()).isSpatiallyEqual(where)
     * @pre (dst_data.getDim() == where.getDim()) &&
     *      (dst_data.getDim() == src_data_old.getDim()) &&
     *      (dst_data.getDim() == src_data_new.getDim())
     * @pre direction vector of dst_data is smaller than that of src_data_old or
     *      src_data_new
     */
    void timeInterpolate(SAMRAI::hier::PatchData &dst_data, const SAMRAI::hier::Box &where,
                         const SAMRAI::hier::PatchData &src_data_old,
                         const SAMRAI::hier::PatchData &src_data_new) const {
        UNIMPLEMENTED;
    }
};

struct spParticlePatchDataFactory : public SAMRAI::hier::PatchDataFactory {
    spParticlePatchDataFactory(int depth, const SAMRAI::hier::IntVector &ghosts,
                               bool fine_boundary_represents_var = true)
        : SAMRAI::hier::PatchDataFactory(ghosts),
          m_depth_(depth),
          m_fine_boundary_represents_var_(fine_boundary_represents_var) {}
    ~spParticlePatchDataFactory() override {}

    virtual std::shared_ptr<SAMRAI::hier::PatchDataFactory> cloneFactory(
        const SAMRAI::hier::IntVector &ghosts) override {
        return std::make_shared<spParticlePatchDataFactory>(getDepth(), ghosts, fineBoundaryRepresentsVariable());
    };

    virtual std::shared_ptr<SAMRAI::hier::PatchData> allocate(const SAMRAI::hier::Patch &patch) const override {
        return std::make_shared<spParticlePatchData>(patch.getBox(), getDepth(), getGhostCellWidth());
    };

    virtual std::shared_ptr<SAMRAI::hier::BoxGeometry> getBoxGeometry(const SAMRAI::hier::Box &box) const override {
        return std::make_shared<SAMRAI::pdat::CellGeometry>(box, d_ghosts);
    };
    int getDepth() const { return m_depth_; }

    virtual size_t getSizeOfMemory(const SAMRAI::hier::Box &box) const override { return 0; };

    virtual bool fineBoundaryRepresentsVariable() const override { return m_fine_boundary_represents_var_; };

    virtual bool dataLivesOnPatchBorder() const override { return false; };

    virtual bool validCopyTo(const std::shared_ptr<SAMRAI::hier::PatchDataFactory> &dst_pdf) const override {
        return true;
    };

   private:
    int m_depth_ = 0;
    int m_pic_ = 100;
    bool m_fine_boundary_represents_var_ = false;
};

struct spParticleVariable : public SAMRAI::hier::Variable {
   public:
    spParticleVariable(const SAMRAI::tbox::Dimension &dim, const std::string &name, int depth = 1)
        : SAMRAI::hier::Variable(
              name, std::make_shared<spParticlePatchDataFactory>(depth, SAMRAI::hier::IntVector::getZero(dim))),
          m_dof_(depth){};

    ~spParticleVariable() override {}

    bool fineBoundaryRepresentsVariable() const override { return false; }

    bool dataLivesOnPatchBorder() const override { return false; }

    int getDepth() const { return m_dof_; }

   private:
    int m_dof_ = 0;
};

struct spParticleVisDerivedDataStrategy : public SAMRAI::appu::VisDerivedDataStrategy {
    spParticleVisDerivedDataStrategy() {}
    ~spParticleVisDerivedDataStrategy() override {}
    bool packDerivedDataIntoDoubleBuffer(double *buffer, const SAMRAI::hier::Patch &patch,
                                         const SAMRAI::hier::Box &region, const std::string &variable_name,
                                         int depth_index, double simulation_time) const override {
        UNIMPLEMENTED;
        return true;
    };
    bool packMixedDerivedDataIntoDoubleBuffer(double *buffer, std::vector<double> &mixbuffer,
                                              const SAMRAI::hier::Patch &patch, const SAMRAI::hier::Box &region,
                                              const std::string &variable_name, int depth_index) const override {
        UNIMPLEMENTED;
        return true;
    };
};

namespace traits {
template <typename TV, int IFORM>
struct SAMRAIDataTraits {
    typedef std::conditional_t<IFORM == VERTEX, SAMRAI::pdat::NodeData<TV>,                                      //
                               std::conditional_t<IFORM == EDGE, SAMRAI::pdat::EdgeData<TV>,                     //
                                                  std::conditional_t<IFORM == FACE, SAMRAI::pdat::SideData<TV>,  //
                                                                     SAMRAI::pdat::CellData<TV>>>>
        data_type;
};
}  // namespace traits {

template <typename TV, int IFORM>
struct spFieldPatchData : public traits::SAMRAIDataTraits<TV, IFORM>::data_type {
    typedef typename traits::SAMRAIDataTraits<TV, IFORM>::data_type base_type;

    explicit spFieldPatchData(const SAMRAI::hier::Box &box, int depth, const SAMRAI::hier::IntVector &ghosts)
        : base_type(box, depth, ghosts) {}

    ~spFieldPatchData() override = default;

    std::shared_ptr<data::DataBlock> GetDataBlock() const { return m_data_block_; }
    void SetDataBlock(std::shared_ptr<data::DataBlock> const &blk) { m_data_block_ = blk; }

   private:
    std::shared_ptr<data::DataBlock> m_data_block_;
};

template <typename TV, int IFORM>
struct spFieldPatchDataFactory : public SAMRAI::hier::PatchDataFactory {
    spFieldPatchDataFactory(int depth, const SAMRAI::hier::IntVector &ghosts, bool fine_boundary_represents_var)
        : SAMRAI::hier::PatchDataFactory(ghosts),
          m_dof_(depth),
          m_fine_boundary_represents_var_(fine_boundary_represents_var) {}
    ~spFieldPatchDataFactory() override {}

    virtual std::shared_ptr<SAMRAI::hier::PatchDataFactory> cloneFactory(
        const SAMRAI::hier::IntVector &ghosts) override {
        return std::make_shared<spFieldPatchDataFactory<TV, IFORM>>(m_dof_, ghosts, m_fine_boundary_represents_var_);
    };

    virtual std::shared_ptr<SAMRAI::hier::PatchData> allocate(const SAMRAI::hier::Patch &patch) const override {
        return std::make_shared<spFieldPatchData<TV, IFORM>>(patch.getBox(), m_dof_, getGhostCellWidth());
    };

    virtual std::shared_ptr<SAMRAI::hier::BoxGeometry> getBoxGeometry(const SAMRAI::hier::Box &box) const override {
        return std::make_shared<SAMRAI::pdat::CellGeometry>(box, d_ghosts);
    };

    virtual size_t getSizeOfMemory(const SAMRAI::hier::Box &box) const override {
        const size_t obj = SAMRAI::tbox::MemoryUtilities::align(sizeof(spFieldPatchData<TV, IFORM>));
        const size_t data = spFieldPatchData<TV, IFORM>::getSizeOfData(box, m_dof_, d_ghosts);
        return obj + data;
    };

    virtual bool fineBoundaryRepresentsVariable() const override { return m_fine_boundary_represents_var_; };

    virtual bool dataLivesOnPatchBorder() const override { return false; };

    virtual bool validCopyTo(const std::shared_ptr<SAMRAI::hier::PatchDataFactory> &dst_pdf) const override {
        bool valid_copy = false;

        /*
         * Valid options are NodeData and OuternodeData.
         */
        if (!valid_copy) {
            std::shared_ptr<spFieldPatchDataFactory<TV, IFORM>> ndf(
                std::dynamic_pointer_cast<spFieldPatchDataFactory<TV, IFORM>, SAMRAI::hier::PatchDataFactory>(dst_pdf));
            if (ndf) { valid_copy = true; }
        }

        //        if (!valid_copy) {
        //            std::shared_ptr<OuternodeDataFactory<TYPE>> ondf(
        //                std::dynamic_pointer_cast<OuternodeDataFactory<TYPE>, hier::PatchDataFactory>(dst_pdf));
        //            if (ondf) { valid_copy = true; }
        //        }

        return valid_copy;
    };

   private:
    int m_dof_ = 0;
    bool m_fine_boundary_represents_var_ = true;
};

template <typename TV, int IFORM>
struct spFieldVariable : public SAMRAI::hier::Variable {
   public:
    spFieldVariable(const SAMRAI::tbox::Dimension &dim, const std::string &name, int depth = 1,
                    bool fine_boundary_represents_var = true)
        : SAMRAI::hier::Variable(
              name, std::make_shared<spFieldPatchDataFactory<TV, IFORM>>(depth, SAMRAI::hier::IntVector::getZero(dim))),
          m_dof_(depth),
          m_fine_boundary_represents_var_(fine_boundary_represents_var){};

    ~spFieldVariable() override {}

    bool fineBoundaryRepresentsVariable() const override { return false; }

    bool dataLivesOnPatchBorder() const override { return false; }

    int getDepth() const { return m_dof_; }

   private:
    int m_dof_ = 0;
    bool m_fine_boundary_represents_var_ = false;
};

REGISTER_CREATOR(SAMRAITimeIntegrator, SAMRAITimeIntegrator)

class SAMRAIHyperbolicPatchStrategyAdapter : public SAMRAI::algs::HyperbolicPatchStrategy {
    SP_OBJECT_BASE(SAMRAIHyperbolicPatchStrategyAdapter)
   public:
    SAMRAIHyperbolicPatchStrategyAdapter(engine::Context *ctx, engine::Atlas *atlas,
                                         std::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom);

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
                                         const std::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
                                         const std::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse,
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
    void registerVisItDataWriter(std::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer);

    /**Print all data members for SAMRAIWorkerHyperbolic class.     */
    void printClassData(std::ostream &os) const;

    //    void Dispatch(SAMRAI::hier::Patch &patch);

    engine::Context *GetContext() { return m_ctx_; }
    engine::Context const *GetContext() const { return m_ctx_; }

    engine::Atlas *GetAtlas() { return m_atlas_; }
    engine::Atlas const *GetAtlas() const { return m_atlas_; }

   private:
    static constexpr int NDIMS = 3;
    engine::Context *m_ctx_;
    engine::Atlas *m_atlas_;
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
    std::shared_ptr<SAMRAI::geom::CartesianGridGeometry> d_grid_geometry = nullptr;
    std::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer = nullptr;

    spParticleVisDerivedDataStrategy d_particle_writer_;
    /*
     * Data items used for nonuniform Load balance, if used.
     */
    std::shared_ptr<SAMRAI::pdat::CellVariable<double>> d_workload_variable;
    int d_workload_data_id = 0;
    bool d_use_nonuniform_workload;
    std::map<id_type, std::pair<engine::AttributeDesc, std::shared_ptr<SAMRAI::hier::Variable>>> m_samrai_variables_;
    SAMRAI::hier::IntVector d_nghosts;
    SAMRAI::hier::IntVector d_fluxghosts;

    engine::Patch *GetPatch(SAMRAI::hier::Patch &patch);
    void PopPatch(SAMRAI::hier::Patch &patch, engine::Patch *p);
    void PushPatch(engine::Patch *p, SAMRAI::hier::Patch &patch);
};

SAMRAIHyperbolicPatchStrategyAdapter::SAMRAIHyperbolicPatchStrategyAdapter(
    engine::Context *ctx, engine::Atlas *atlas, std::shared_ptr<SAMRAI::geom::CartesianGridGeometry> const &grid_geom)
    : d_dim(3),
      d_grid_geometry(grid_geom),
      d_use_nonuniform_workload(false),
      d_nghosts(d_dim, 4),
      d_fluxghosts(d_dim, 1),
      m_ctx_(ctx),
      m_atlas_(atlas) {
    TBOX_ASSERT(grid_geom);
}

SAMRAIHyperbolicPatchStrategyAdapter::~SAMRAIHyperbolicPatchStrategyAdapter() = default;

namespace detail {
template <typename T>
std::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable_t(const engine::AttributeDesc &attr) {
    SAMRAI::tbox::Dimension d_dim(3);

    std::shared_ptr<SAMRAI::hier::Variable> res;
    switch (attr.GetIFORM()) {
        case VERTEX:
            res = std::make_shared<SAMRAI::pdat::NodeVariable<T>>(d_dim, attr.GetPrefix(), attr.GetDOF());
            break;
        case EDGE:
            res = std::make_shared<SAMRAI::pdat::EdgeVariable<T>>(d_dim, attr.GetPrefix(), attr.GetDOF());
            break;
        case FACE:
            res = std::make_shared<SAMRAI::pdat::SideVariable<T>>(d_dim, attr.GetPrefix(), attr.GetDOF());
            break;
        case VOLUME:
            res = std::make_shared<SAMRAI::pdat::CellVariable<T>>(d_dim, attr.GetPrefix(), attr.GetDOF());
            break;
        default:
            break;
    }
    return res;
}

std::shared_ptr<SAMRAI::hier::Variable> create_samrai_variable(const engine::AttributeDesc &attr) {
    std::shared_ptr<SAMRAI::hier::Variable> res = nullptr;
    SAMRAI::tbox::Dimension d_dim(3);
    if (attr.GetIFORM() == FIBER) {
        res = std::make_shared<spParticleVariable>(d_dim, attr.GetPrefix(), attr.GetDOF());
    } else if (attr.value_type_info() == (typeid(float))) {
        res = create_samrai_variable_t<float>(attr);
    } else if (attr.value_type_info() == (typeid(double))) {
        res = create_samrai_variable_t<double>(attr);
    } else if (attr.value_type_info() == (typeid(int))) {
        res = create_samrai_variable_t<int>(attr);
    } else {
        RUNTIME_ERROR << " attr [" << attr.GetPrefix() << "] is not supported!" << std::endl;
    }
    return res;
}

template <typename T, int NDIMS>
Array<T, ZSFC<NDIMS>> create_array(SAMRAI::pdat::ArrayData<T> &p_data, int depth = 0) {
    auto i_lower = p_data.getBox().lower();
    auto i_upper = p_data.getBox().upper();

    return Array<T, ZSFC<NDIMS>>(
        p_data.getPointer(depth),
        index_box_type{{i_lower[0], i_lower[1], i_lower[2]}, {i_upper[0] + 1, i_upper[1] + 1, i_upper[2] + 1}}, true);
    ;
};

template <int NDIMS, typename T>
std::shared_ptr<data::DataBlock> create_simpla_datablock(int IFORM, std::shared_ptr<SAMRAI::hier::PatchData> pd) {
    typedef Array<T, ZSFC<NDIMS>> array_type;
    std::shared_ptr<data::DataBlock> res = nullptr;

    switch (IFORM) {
        case VERTEX: {
            auto p_data = std::dynamic_pointer_cast<SAMRAI::pdat::NodeData<T>>(pd);
            int depth = p_data->getDepth();
            auto mArray = std::make_shared<data::DataMultiArray<array_type>>(depth);
            for (int d = 0; d < depth; ++d) {
                create_array<T, NDIMS>(p_data->getArrayData(), d).swap(mArray->GetArray(d));
            }

            res = std::dynamic_pointer_cast<data::DataBlock>(mArray);
            break;
        }
        case EDGE: {
            auto p_data = std::dynamic_pointer_cast<SAMRAI::pdat::EdgeData<T>>(pd);
            int depth = p_data->getDepth();
            auto mArray = std::make_shared<data::DataMultiArray<array_type>>(depth * 3);
            for (int axis = 0; axis < 3; ++axis) {
                for (int d = 0; d < depth; ++d) {
                    create_array<T, NDIMS>(p_data->getArrayData(axis), d).swap(mArray->GetArray(axis * depth + d));
                }
            }
            res = std::dynamic_pointer_cast<data::DataBlock>(mArray);
            break;
        }
        case FACE: {
            auto p_data = std::dynamic_pointer_cast<SAMRAI::pdat::SideData<T>>(pd);
            int depth = p_data->getDepth();
            auto mArray = std::make_shared<data::DataMultiArray<array_type>>(depth * 3);
            for (int axis = 0; axis < 3; ++axis) {
                for (int d = 0; d < depth; ++d) {
                    create_array<T, NDIMS>(p_data->getArrayData(axis), d).swap(mArray->GetArray(axis * depth + d));
                }
            }
            res = std::dynamic_pointer_cast<data::DataBlock>(mArray);
            break;
        }
        case VOLUME: {
            auto p_data = std::dynamic_pointer_cast<SAMRAI::pdat::CellData<T>>(pd);
            int depth = p_data->getDepth();
            auto mArray = std::make_shared<data::DataMultiArray<array_type>>(depth);
            for (int d = 0; d < depth; ++d) {
                create_array<T, NDIMS>(p_data->getArrayData(), d).swap(mArray->GetArray(d));
            }
            res = std::dynamic_pointer_cast<data::DataBlock>(mArray);
            break;
        }
        case FIBER: {
            auto p_data = std::dynamic_pointer_cast<spParticlePatchData>(pd);
            ASSERT(p_data != nullptr);
            res = p_data->GetDataBlock();
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
std::shared_ptr<data::DataBlock> create_simpla_datablock(const engine::AttributeDesc &desc,
                                                         std::shared_ptr<SAMRAI::hier::PatchData> pd) {
    std::shared_ptr<data::DataBlock> res(nullptr);
    if (desc.value_type_info() == (typeid(float))) {
        res = create_simpla_datablock<NDIMS, float>(desc.GetIFORM(), pd);
    } else if (desc.value_type_info() == (typeid(double))) {
        res = create_simpla_datablock<NDIMS, double>(desc.GetIFORM(), pd);
    } else if (desc.value_type_info() == (typeid(int))) {
        res = create_simpla_datablock<NDIMS, int>(desc.GetIFORM(), pd);
    } else {
        RUNTIME_ERROR << "Unsupported m_value_ value_type_info" << std::endl;
    }
    return res;
}
std::shared_ptr<SAMRAI::hier::PatchData> convert_from_data_block(const engine::AttributeDesc &desc,
                                                                 std::shared_ptr<data::DataBlock> t) {
    //    UNIMPLEMENTED;
    return nullptr;
}

}  // namespace detail

void SAMRAIHyperbolicPatchStrategyAdapter::registerModelVariables(SAMRAI::algs::HyperbolicLevelIntegrator *integrator) {
    ASSERT(integrator != nullptr);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();

    SAMRAI::hier::IntVector d_nghosts{d_dim, 4};
    SAMRAI::hier::IntVector d_fluxghosts{d_dim, 1};
    //**************************************************************

    for (auto const *item : m_ctx_->GetMesh()->GetAttributes()) {
        if (item->db()->Check("IS_NOT_OWNED", true) ||
            m_samrai_variables_.find(item->GetDescID()) != m_samrai_variables_.end()) {
            continue;
        }

        auto var = simpla::detail::create_samrai_variable(item->GetDescription());

        m_samrai_variables_.emplace(item->GetDescID(), std::make_pair(engine::AttributeDesc(*item), var));

        /*** NOTE:
        *  1. SAMRAI Visit Writer only support NODE and CELL variable (double,float ,int)
        *  2. SAMRAI   SAMRAI::algs::HyperbolicLevelIntegrator->registerVariable only support double
        **/
        SAMRAI::algs::HyperbolicLevelIntegrator::HYP_VAR_TYPE v_type =
            SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP;
        SAMRAI::hier::IntVector ghosts = d_nghosts;
        std::string coarsen_name = "";
        std::string refine_name = "";

        if (item->db()->Check("COORDINATES", true)) { v_type = SAMRAI::algs::HyperbolicLevelIntegrator::INPUT; }
        if (item->db()->Check("INPUT", true)) { v_type = SAMRAI::algs::HyperbolicLevelIntegrator::INPUT; }
        if (item->db()->Check("FLUX", true)) {
            ghosts = d_fluxghosts;
            v_type = SAMRAI::algs::HyperbolicLevelIntegrator::FLUX;
            coarsen_name = "CONSERVATIVE_COARSEN";
            refine_name = "";
        }
        if (item->GetIFORM() == FIBER) {
            v_type = SAMRAI::algs::HyperbolicLevelIntegrator::TIME_DEP;
            coarsen_name = "CONSTANT_COARSEN";
            refine_name = "NO_REFINE";
        }

        //        if ((item->GetTypeInfo() != typeid(double)) || item->db()->Check("TEMP", true)) {
        //            v_type = SAMRAI::algs::HyperbolicLevelIntegrator::TEMPORARY;
        //            coarsen_name = "";
        //            refine_name = "";
        //        }

        integrator->registerVariable(var, ghosts, v_type, d_grid_geometry, coarsen_name, refine_name);

        std::string visit_variable_type = "SCALAR";
        ;
        if ((item->GetIFORM() == VERTEX || item->GetIFORM() == VOLUME) && (item->GetDOF() == 1)) {
            visit_variable_type = "SCALAR";
        } else if (((item->GetIFORM() == EDGE || item->GetIFORM() == FACE) && (item->GetDOF() == 1)) ||
                   ((item->GetIFORM() == VERTEX || item->GetIFORM() == VOLUME) && (item->GetDOF() == 3))) {
            visit_variable_type = "VECTOR";
        } else if (((item->GetIFORM() == VERTEX || item->GetIFORM() == VOLUME) && item->GetDOF() == 9) ||
                   ((item->GetIFORM() == EDGE || item->GetIFORM() == FACE) && item->GetDOF() == 3)) {
            visit_variable_type = "TENSOR";
        }

        //        else {
        //            VERBOSE << "Can not register attribute [" << item->GetName() << ":" << item->GetFancyTypeName()
        //                    << "] to VisIt writer !" << std::endl;
        //        }
        //        v_type != SAMRAI::algs::HyperbolicLevelIntegrator::TEMPORARY
        if (visit_variable_type != "" && item->db()->Check("COORDINATES", true)) {
            d_visit_writer->registerNodeCoordinates(
                vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
        } else if ((item->GetIFORM() == VERTEX || item->GetIFORM() == VOLUME)) {
            d_visit_writer->registerPlotQuantity(
                item->GetPrefix(), visit_variable_type,
                vardb->mapVariableAndContextToIndex(var, integrator->getPlotContext()));
        } else {
            //            d_visit_writer->registerDerivedPlotQuantity(item->GetPrefix(), visit_variable_type,
            //            &d_particle_writer_,
            //                                                        1.0, "CELL", "CLEAN");
        }
    }
    //    integrator->printClassData(std::cout);
    //    vardb->printClassData(std::cout);
}

engine::Patch *SAMRAIHyperbolicPatchStrategyAdapter::GetPatch(SAMRAI::hier::Patch &patch) {
    return GetAtlas()->GetPatch(engine::MeshBlock{
        index_box_type{{patch.getBox().lower()[0], patch.getBox().lower()[1], patch.getBox().lower()[2]},
                       {patch.getBox().upper()[0] + 1, patch.getBox().upper()[1] + 1, patch.getBox().upper()[2] + 1}},
        patch.getLocalId().getValue(), patch.getPatchLevelNumber(), patch.getGlobalId().getOwnerRank()});
}
void SAMRAIHyperbolicPatchStrategyAdapter::PopPatch(SAMRAI::hier::Patch &patch, engine::Patch *p) {
    for (auto &item : m_samrai_variables_) {
        auto samrai_id = SAMRAI::hier::VariableDatabase::getDatabase()->mapVariableAndContextToIndex(item.second.second,
                                                                                                     getDataContext());

        if (!patch.checkAllocated(samrai_id)) { patch.allocatePatchData(samrai_id); }

        p->SetDataBlock(item.first, simpla::detail::create_simpla_datablock<NDIMS>(item.second.first,
                                                                                   patch.getPatchData(samrai_id)));
    }
}
void SAMRAIHyperbolicPatchStrategyAdapter::PushPatch(engine::Patch *p, SAMRAI::hier::Patch &patch) {}

void SAMRAIHyperbolicPatchStrategyAdapter::setupLoadBalancer(SAMRAI::algs::HyperbolicLevelIntegrator *integrator,
                                                             SAMRAI::mesh::GriddingAlgorithm *gridding_algorithm) {
    const SAMRAI::hier::IntVector &zero_vec = SAMRAI::hier::IntVector::getZero(d_dim);
    SAMRAI::hier::VariableDatabase *vardb = SAMRAI::hier::VariableDatabase::getDatabase();
    if (d_use_nonuniform_workload && (gridding_algorithm != nullptr)) {
        auto load_balancer =
            std::dynamic_pointer_cast<SAMRAI::mesh::CascadePartitioner>(gridding_algorithm->getLoadBalanceStrategy());
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

void SAMRAIHyperbolicPatchStrategyAdapter::initializeDataOnPatch(SAMRAI::hier::Patch &patch, double data_time,
                                                                 bool initial_time) {
    if (initial_time) {
        simpla::engine::Patch *p = GetPatch(patch);
        PopPatch(patch, p);

        index_tuple gw{4, 4, 4};  // = p.GetMeshBlock().GetGhostWidth();

        auto pgeom = std::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());

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
                //                CHECK(p->GetMeshBlock()->IndexBox());
                //                CHECK(vertex_box);
                //                CHECK(edge0_box);
                //                CHECK(edge1_box);
                //                CHECK(edge2_box);
                //                CHECK(face0_box);
                //                CHECK(face1_box);
                //                CHECK(face2_box);
                //                CHECK(volume_box);

                auto &d = *m_ctx_->GetMesh();
                d.Push(p);
                d.GetRange("PATCH_BOUNDARY_" + std::to_string(VERTEX))
                    .append(std::make_shared<ContinueRange<EntityId>>(vertex_box, 0));

                d.GetRange("PATCH_BOUNDARY_" + std::to_string(EDGE))
                    .append(std::make_shared<ContinueRange<EntityId>>(edge0_box, 1))
                    .append(std::make_shared<ContinueRange<EntityId>>(edge1_box, 2))
                    .append(std::make_shared<ContinueRange<EntityId>>(edge2_box, 4));

                d.GetRange("PATCH_BOUNDARY_" + std::to_string(FACE))
                    .append(std::make_shared<ContinueRange<EntityId>>(face0_box, 6))
                    .append(std::make_shared<ContinueRange<EntityId>>(face1_box, 5))
                    .append(std::make_shared<ContinueRange<EntityId>>(face2_box, 3));

                d.GetRange("PATCH_BOUNDARY_" + std::to_string(VOLUME))
                    .append(std::make_shared<ContinueRange<EntityId>>(volume_box, 7));
                d.Pop(p);
            }
        //        m_ctx_->InitialCondition(&p, data_time);
        m_ctx_->Push(p);
        m_ctx_->InitialCondition(data_time);
        m_ctx_->Pop(p);

        PushPatch(p, patch);

        //        m_ctx_->GetBaseMesh()->Deserialize(p.get());
        //        VERBOSE << "DoInitialize MeshBase : " << m_ctx_->GetBaseMesh()->GetRegisterName() <<
        //        std::endl;
        //        m_ctx_->GetBaseMesh()->InitialCondition(data_time);
        //        for (auto const &item : m_ctx_->GetModel().GetAll()) {
        //            m_ctx_->GetBaseMesh()->RegisterRanges(item.second, item.first);
        //        }
        //
        //        for (auto &d : m_ctx_->GetAllDomains()) {
        //            VERBOSE << "DoInitialize DomainBase : " << d.first << std::endl;
        //            d.second->InitialCondition(p.get(), data_time);
        //        }
        //        m_ctx_->GetBaseMesh()->Serialize(p.get());
    }

    if (d_use_nonuniform_workload) {
        if (!patch.checkAllocated(d_workload_data_id)) { patch.allocatePatchData(d_workload_data_id); }
        std::shared_ptr<SAMRAI::pdat::CellData<double>> workload_data(
            std::dynamic_pointer_cast<SAMRAI::pdat::CellData<double>, SAMRAI::hier::PatchData>(
                patch.getPatchData(d_workload_data_id)));
        TBOX_ASSERT(workload_data);

        const SAMRAI::hier::Box &box = patch.getBox();
        const SAMRAI::hier::BoxId &box_id = box.getBoxId();
        const SAMRAI::hier::LocalId &local_id = box_id.getLocalId();
        double id_val = (local_id.getValue() % 2 == 0) ? static_cast<double>(local_id.getValue() % 10) : 0.0;
        workload_data->fillAll(1.0 + id_val);
    }
}

double SAMRAIHyperbolicPatchStrategyAdapter::computeStableDtOnPatch(SAMRAI::hier::Patch &patch, bool initial_time,
                                                                    double dt_time) {
    //    auto pgeom =
    //    std::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(patch.getPatchGeometry());
    //    return pgeom->getDx()[0] / 2.0;
    return dt_time;
}

void SAMRAIHyperbolicPatchStrategyAdapter::computeFluxesOnPatch(SAMRAI::hier::Patch &patch, double time_now,
                                                                double time_dt) {
    auto *p = GetPatch(patch);
    PopPatch(patch, p);
    m_ctx_->Push(p);
    m_ctx_->ComputeFluxes(time_now, time_dt);
    m_ctx_->Pop(p);
    PushPatch(p, patch);
}

void SAMRAIHyperbolicPatchStrategyAdapter::conservativeDifferenceOnPatch(SAMRAI::hier::Patch &patch, double time_now,
                                                                         double time_dt, bool at_syncronization) {
    auto *p = GetPatch(patch);
    PopPatch(patch, p);
    m_ctx_->Push(p);
    m_ctx_->Advance(time_now, time_dt);
    m_ctx_->Pop(p);
    PushPatch(p, patch);
}

void SAMRAIHyperbolicPatchStrategyAdapter::tagRichardsonExtrapolationCells(
    SAMRAI::hier::Patch &patch, const int error_level_number,
    const std::shared_ptr<SAMRAI::hier::VariableContext> &coarsened_fine,
    const std::shared_ptr<SAMRAI::hier::VariableContext> &advanced_coarse, double regrid_time, double deltat,
    const int error_coarsen_ratio, bool initial_error, const int tag_index, bool uses_gradient_detector_too) {}

void SAMRAIHyperbolicPatchStrategyAdapter::tagGradientDetectorCells(SAMRAI::hier::Patch &patch, double regrid_time,
                                                                    bool initial_error, int tag_index,
                                                                    bool uses_richardson_extrapolation_too) {
    NULL_USE(regrid_time);
    NULL_USE(initial_error);
    NULL_USE(uses_richardson_extrapolation_too);

    auto *p = GetPatch(patch);
    PopPatch(patch, p);
    auto desc = m_ctx_->GetMesh()->GetAttributeDescription("_refinement_tags_");
    if (desc != nullptr) {
        p->SetDataBlock(desc->GetDescID(),
                        simpla::detail::create_simpla_datablock<NDIMS>(*desc, patch.getPatchData(tag_index)));
        m_ctx_->Push(p);
        m_ctx_->TagRefinementCells(regrid_time);
        m_ctx_->Pop(p);
    }
    PushPatch(p, patch);
}

void SAMRAIHyperbolicPatchStrategyAdapter::setPhysicalBoundaryConditions(
    SAMRAI::hier::Patch &patch, double fill_time, const SAMRAI::hier::IntVector &ghost_width_to_fill) {
    auto *p = GetPatch(patch);
    PopPatch(patch, p);
    m_ctx_->Push(p);
    m_ctx_->BoundaryCondition(fill_time, 0);
    m_ctx_->Pop(p);
    PushPatch(p, patch);
}

void SAMRAIHyperbolicPatchStrategyAdapter::registerVisItDataWriter(
    std::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer) {
    TBOX_ASSERT(viz_writer);
    d_visit_writer = viz_writer;
}

void SAMRAIHyperbolicPatchStrategyAdapter::printClassData(std::ostream &os) const {
    os << "\nSAMRAIWorkerAdapter::printClassData..." << std::endl;
    os << "m_domain_geo_prefix_ = " << m_name_ << std::endl;
    os << "d_grid_geometry = " << d_grid_geometry.get() << std::endl;
    os << std::endl;
}

struct SAMRAITimeIntegrator::pimpl_s {
    std::shared_ptr<SAMRAIHyperbolicPatchStrategyAdapter> hyperbolic_patch_strategy;
    std::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geometry;
    std::shared_ptr<SAMRAI::hier::PatchHierarchy> patch_hierarchy;
    std::shared_ptr<SAMRAI::algs::HyperbolicLevelIntegrator> hyp_level_integrator;

    //    std::shared_ptr<SAMRAI::engine::StandardTagAndInitialize> error_detector;
    //    std::shared_ptr<SAMRAI::engine::BergerRigoutsos> box_generator;
    //    std::shared_ptr<SAMRAI::engine::CascadePartitioner> load_balancer;
    //    std::shared_ptr<SAMRAI::engine::GriddingAlgorithm> gridding_algorithm;
    std::shared_ptr<SAMRAI::algs::TimeRefinementIntegrator> m_time_refinement_integrator_;
    // VisItDataWriter is only present if HDF is available
    std::shared_ptr<SAMRAI::appu::VisItDataWriter> visit_data_writer_;

    bool write_restart = false;
    int restart_interval = 0;

    std::string restart_write_dirname;

    bool viz_dump_data = false;
    int viz_dump_interval = 1;

    unsigned int ndims = 3;

    std::string m_output_URL_ = "";
};

SAMRAITimeIntegrator::SAMRAITimeIntegrator() : m_pimpl_(new pimpl_s) {}

SAMRAITimeIntegrator::~SAMRAITimeIntegrator() {
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
}

void SAMRAITimeIntegrator::Synchronize() { engine::TimeIntegrator::Synchronize(); }

std::shared_ptr<data::DataTable> SAMRAITimeIntegrator::Serialize() const { return base_type::Serialize(); }

void SAMRAITimeIntegrator::Deserialize(const std::shared_ptr<data::DataTable> &cfg) {
    base_type::Deserialize(cfg);
    m_pimpl_->m_output_URL_ = cfg->GetValue<std::string>("OutputURL", GetName() + ".simpla");
}

void SAMRAITimeIntegrator::DoInitialize() {
    dcomplex a = std::numeric_limits<dcomplex>::signaling_NaN();
    engine::TimeIntegrator::DoInitialize();
    /** Setup SAMRAI::tbox::MPI.      */
    SAMRAI::tbox::SAMRAI_MPI::init(*reinterpret_cast<MPI_Comm const *>(GLOBAL_COMM.comm()));  //
    SAMRAI::tbox::SAMRAIManager::initialize();
    /** Setup SAMRAI, enable logging, and process command line.     */
    SAMRAI::tbox::SAMRAIManager::startup();

    //    data::DataTable(std::make_shared<DataBackendSAMRAI>()).swap(*db());
    //    const SAMRAI::tbox::SAMRAI_MPI & mpi(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());
}

void SAMRAITimeIntegrator::DoTearDown() { engine::TimeIntegrator::DoTearDown(); }

void SAMRAITimeIntegrator::DoUpdate() {
    engine::TimeIntegrator::DoUpdate();
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

    unsigned int ndims = 3;                // static_cast<unsigned int>( GetContext()->GetNDims());
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

    auto cfgCartesianGridGeometry = std::make_shared<SAMRAI::tbox::MemoryDatabase>("CartesianGeometry");

    nTuple<int, 3> i_low{0, 0, 0};
    nTuple<int, 3> i_up{0, 0, 0};

    std::tie(i_low, i_up) = GetContext()->IndexBox();

    cfgCartesianGridGeometry->putDatabaseBox(
        "domain_boxes_0", SAMRAI::tbox::DatabaseBox{SAMRAI::tbox::Dimension(3), &i_low[0], &i_up[0]});

    cfgCartesianGridGeometry->putIntegerArray("periodic_dimension", &GetAtlas()->GetPeriodicDimensions()[0], ndims);

    auto x_box = GetContext()->GetMesh()->GetBox(0);
    cfgCartesianGridGeometry->putDoubleArray("x_lo", &std::get<0>(x_box)[0], ndims);
    cfgCartesianGridGeometry->putDoubleArray("x_up", &std::get<1>(x_box)[0], ndims);

    m_pimpl_->grid_geometry.reset(
        new SAMRAI::geom::CartesianGridGeometry(dim, "CartesianGeometry", cfgCartesianGridGeometry));

    m_pimpl_->grid_geometry->addCoarsenOperator(typeid(spParticleVariable).name(),
                                                std::make_shared<spParticleCoarsenRefine>());

    m_pimpl_->grid_geometry->addRefineOperator(typeid(spParticleVariable).name(),
                                               std::make_shared<spParticleConstantRefine>());

    m_pimpl_->grid_geometry->addTimeInterpolateOperator(typeid(spParticleVariable).name(),
                                                        std::make_shared<spParticleLinearTimeInterpolateOp>());
    //---------------------------------

    auto cfgPatchHierarchy = std::make_shared<SAMRAI::tbox::MemoryDatabase>("cfgPatchHierarchy");

    // Maximum number of levels in hierarchy.
    cfgPatchHierarchy->putInteger("max_levels", static_cast<int>(GetAtlas()->GetMaxLevel()));

    auto ratio_to_coarser = cfgPatchHierarchy->putDatabase("ratio_to_coarser");

    for (int i = 0, n = static_cast<int>(GetAtlas()->GetMaxLevel()); i < n; ++i) {
        nTuple<int, 3> level;
        level = GetAtlas()->GetRefineRatio(i);
        ratio_to_coarser->putIntegerArray("level_" + std::to_string(i), &level[0], ndims);
    }

    auto largest_patch_size = cfgPatchHierarchy->putDatabase("largest_patch_size");
    auto smallest_patch_size = cfgPatchHierarchy->putDatabase("smallest_patch_size");

    smallest_patch_size->putIntegerArray("level_0", &GetAtlas()->GetSmallestPatchDimensions()[0], ndims);
    largest_patch_size->putIntegerArray("level_0", &GetAtlas()->GetLargestPatchDimensions()[0], ndims);

    m_pimpl_->patch_hierarchy.reset(
        new SAMRAI::hier::PatchHierarchy("cfgPatchHierarchy", m_pimpl_->grid_geometry, cfgPatchHierarchy));

    auto cfgHyperbolicLevelIntegrator = std::make_shared<SAMRAI::tbox::MemoryDatabase>("cfgHyperbolicLevelIntegrator");

    // Refer to algs::HyperbolicLevelIntegrator for input
    // max cfl factor used in problem
    cfgHyperbolicLevelIntegrator->putDouble("cfl", engine::TimeIntegrator::GetCFL());
    cfgHyperbolicLevelIntegrator->putDouble("cfl_init", 0.9);
    cfgHyperbolicLevelIntegrator->putBool("lag_dt_computation", true);
    cfgHyperbolicLevelIntegrator->putBool("use_ghosts_to_compute_dt", true);

    /**
     *  create m_pimpl_->hyp_level_integrator and error_detector
     */
    m_pimpl_->hyperbolic_patch_strategy.reset(
        new SAMRAIHyperbolicPatchStrategyAdapter(GetContext(), GetAtlas(), m_pimpl_->grid_geometry));

    m_pimpl_->hyp_level_integrator.reset(new SAMRAI::algs::HyperbolicLevelIntegrator(
        "SAMRAILevelIntegrator", cfgHyperbolicLevelIntegrator, m_pimpl_->hyperbolic_patch_strategy.get(),
        use_refined_timestepping));

    auto cfgStandardTagAndInitialize = std::make_shared<SAMRAI::tbox::MemoryDatabase>("StandardTagAndInitialize");
    // Refer to mesh::StandardTagAndInitialize for input
    cfgStandardTagAndInitialize->putString("tagging_method", "GRADIENT_DETECTOR");

    auto error_detector = std::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
        "StandardTagAndInitialize", m_pimpl_->hyp_level_integrator.get(), cfgStandardTagAndInitialize);
    /**
     *  create grid_algorithm
     */
    auto cfgBergerRigoutsos = std::make_shared<SAMRAI::tbox::MemoryDatabase>("BergerRigoutsos");

    cfgBergerRigoutsos->putBool("sort_output_nodes", true);       // Makes results repeatable.
    cfgBergerRigoutsos->putDouble("efficiency_tolerance", 0.85);  // min % of GetTag cells in new patch level,
    cfgBergerRigoutsos->putDouble("combine_efficiency", 0.95);    //  chop box if  sum of volumes of   smaller
    // boxes <  efficiency * vol of large box

    auto box_generator = std::make_shared<SAMRAI::mesh::BergerRigoutsos>(dim, cfgBergerRigoutsos);

    box_generator->useDuplicateMPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());

    auto cfgLoadBalancer = std::make_shared<SAMRAI::tbox::MemoryDatabase>("LoadBalancer");
    auto load_balancer = std::make_shared<SAMRAI::mesh::CascadePartitioner>(dim, "LoadBalancer", cfgLoadBalancer);

    load_balancer->setSAMRAI_MPI(SAMRAI::tbox::SAMRAI_MPI::getSAMRAIWorld());
    //    load_balancer->printStatistics(std::cout);

    auto cfgGriddingAlgorithm = std::make_shared<SAMRAI::tbox::MemoryDatabase>("GriddingAlgorithm");

    auto gridding_algorithm = std::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
        m_pimpl_->patch_hierarchy, "GriddingAlgorithm", cfgGriddingAlgorithm, error_detector, box_generator,
        load_balancer);

    // Refer to algs::TimeRefinementIntegrator for input
    auto cfgTimeRefinementIntegrator = std::make_shared<SAMRAI::tbox::MemoryDatabase>("TimeRefinementIntegrator");

    cfgTimeRefinementIntegrator->putDouble("start_time",
                                           engine::TimeIntegrator::GetTimeNow());  // initial simulation time
    cfgTimeRefinementIntegrator->putDouble("end_time", engine::TimeIntegrator::GetTimeEnd());  // final simulation time
    cfgTimeRefinementIntegrator->putDouble("grow_dt", 1.1);  // growth factor for timesteps
    cfgTimeRefinementIntegrator->putInteger("max_integrator_steps", 100);

    m_pimpl_->m_time_refinement_integrator_.reset(new SAMRAI::algs::TimeRefinementIntegrator(
        "TimeRefinementIntegrator", cfgTimeRefinementIntegrator, m_pimpl_->patch_hierarchy,
        m_pimpl_->hyp_level_integrator, gridding_algorithm));

    m_pimpl_->visit_data_writer_.reset(
        new SAMRAI::appu::VisItDataWriter(dim, "SimPLA VisIt Writer", m_pimpl_->m_output_URL_, 1));

    m_pimpl_->hyperbolic_patch_strategy->registerVisItDataWriter(m_pimpl_->visit_data_writer_);

    // m_pimpl_->grid_geometry->printClassData(std::cout);
    // m_pimpl_->hyp_level_integrator->printClassData(std::cout);

    m_pimpl_->m_time_refinement_integrator_->initializeHierarchy();

    // m_pimpl_->m_time_refinement_integrator_->printClassData(std::cout);

    MESSAGE << "==================  Context is initialized!  =================" << std::endl;
};

void SAMRAITimeIntegrator::DoFinalize() {
    m_pimpl_->visit_data_writer_.reset();
    m_pimpl_->m_time_refinement_integrator_.reset();
    m_pimpl_->hyp_level_integrator.reset();
    m_pimpl_->hyperbolic_patch_strategy.reset();
    engine::TimeIntegrator::DoFinalize();
}

Real SAMRAITimeIntegrator::Advance(Real time_dt) {
    ASSERT(m_pimpl_->m_time_refinement_integrator_ != nullptr);

    // SetTimeNow(m_pack_->m_time_refinement_integrator->getIntegratorTime());
    Real loop_time = GetTimeNow();
    Real loop_time_end = std::min(loop_time + time_dt, GetTimeEnd());
    Real loop_dt = time_dt;
    while ((loop_time < loop_time_end) &&
           (loop_dt > 0)) {  //&& m_pack_->m_time_refinement_integrator->stepsRemaining() > 0
        Real dt_new = m_pimpl_->m_time_refinement_integrator_->advanceHierarchy(loop_dt, false);
        loop_dt = std::min(dt_new, loop_time_end - loop_time);
        loop_time += loop_dt;
    }

    SetTimeNow(loop_time_end);
    return loop_time_end;
}

void SAMRAITimeIntegrator::CheckPoint() const {
    if (m_pimpl_->visit_data_writer_ != nullptr) {
        m_pimpl_->visit_data_writer_->writePlotData(m_pimpl_->patch_hierarchy,
                                                    m_pimpl_->m_time_refinement_integrator_->getIntegratorStep(),
                                                    m_pimpl_->m_time_refinement_integrator_->getIntegratorTime());
    }
}

void SAMRAITimeIntegrator::Dump() const {
    //    if (m_pack_->visit_data_writer != nullptr) {
    //        VERBOSE << "Dump : Step = " << GetNumberOfStep() << std::end;
    //        m_pack_->visit_data_writer->writePlotData(m_pack_->patch_hierarchy,
    //        m_pack_->m_time_refinement_integrator->getIntegratorStep(),
    //                                         m_pack_->m_time_refinement_integrator->getIntegratorTime());
    //    }
}

bool SAMRAITimeIntegrator::Done() const {
    // m_pack_->m_time_refinement_integrator != nullptr ?
    // !m_pack_->m_time_refinement_integrator->stepsRemaining():;
    return engine::TimeIntegrator::Done();
}

}  // namespace simpla
