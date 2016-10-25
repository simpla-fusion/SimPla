//
// Created by salmon on 16-10-25.
//

#ifndef SIMPLA_SAMRAIWORKER_H
#define SIMPLA_SAMRAIWORKER_H

#include <simpla/SIMPLA_config.h>
#include <simpla/simulation/Worker.h>


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

namespace simpla
{
class SAMRAIWorker : public simulation::WorkerBase,
                     public SAMRAI::algs::TimeRefinementLevelStrategy,
                     public SAMRAI::mesh::StandardTagAndInitStrategy,
                     public SAMRAI::tbox::Serializable
{
public:
    SAMRAIWorker(const std::string &object_name,
                 const SAMRAI::tbox::Dimension &dim,
                 boost::shared_ptr<SAMRAI::tbox::Database> input_db,
                 boost::shared_ptr<SAMRAI::geom::CartesianGridGeometry> grid_geom) {}

    virtual ~SAMRAIWorker() {}

    void
    printClassData(std::ostream &os) const {};

    boost::shared_ptr<SAMRAI::appu::VisItDataWriter> d_visit_writer;

    void
    registerVisItDataWriter(boost::shared_ptr<SAMRAI::appu::VisItDataWriter> viz_writer)
    {
        TBOX_ASSERT(viz_writer);
        d_visit_writer = viz_writer;
    };

    /** for Serializable */
    void
    putToRestart(const boost::shared_ptr<SAMRAI::tbox::Database> &restart_db) const {}

    /** @}*/




    /**
     *  for StandardTagAndInitialize
     * @{
     */

    virtual void
    initializeLevelData(
            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
            const int level_number,
            const double init_data_time,
            const bool can_be_refined,
            const bool initial_time,
            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &old_level =
            boost::shared_ptr<SAMRAI::hier::PatchLevel>(),
            const bool allocate_data = true) {};


    virtual void
    resetHierarchyConfiguration(
            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
            const int coarsest_level,
            const int finest_level) {};

    virtual void
    applyGradientDetector(
            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
            const int level_number,
            const double error_data_time,
            const int tag_index,
            const bool initial_time,
            const bool uses_richardson_extrapolation_too) {}
    /** @}*/
    /**TimeRefinementLevelStrategy
     * @{
     **/


    virtual double
    getLevelDt(
            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
            const double dt_time,
            const bool initial_time) { return dt_time; };

    virtual double
    getMaxFinerLevelDt(
            const int finer_level_number,
            const double coarse_dt,
            const SAMRAI::hier::IntVector &ratio) { return coarse_dt; };

    virtual double
    advanceLevel(
            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
            const double current_time,
            const double new_time,
            const bool first_step,
            const bool last_step,
            const bool regrid_advance = false) {};

    virtual void
    resetTimeDependentData(
            const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level,
            const double new_time,
            const bool can_be_refined) {};


//    virtual void
//    resetDataToPreadvanceState(const boost::shared_ptr<SAMRAI::hier::PatchLevel> &level) {};
//
//
//    virtual void
//    standardLevelSynchronization(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int coarsest_level,
//            const int finest_level,
//            const double sync_time,
//            const std::vector<double> &old_times) {};
//
//
//    virtual void
//    synchronizeNewLevels(
//            const boost::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
//            const int coarsest_level,
//            const int finest_level,
//            const double sync_time,
//            const bool initial_time) {};

    virtual bool
    usingRefinedTimestepping() const { return true; };


    virtual void
    initializeLevelIntegrator(
            const boost::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_alg) {};




    /** @} */
};

}
#endif //SIMPLA_SAMRAIWORKER_H
