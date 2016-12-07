//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <vector>
#include <set>
#include <simpla/data/DataEntityTable.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/design_pattern/Observer.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/LifeControllable.h>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/manifold/Atlas.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/model/Model.h>
#include "Chart.h"

namespace simpla { namespace mesh
{
struct MeshBlock;
struct DataBlock;

struct Chart;
struct CoordinateFrame;

class Worker :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable,
        public concept::LifeControllable
{
public:
    SP_OBJECT_HEAD(Worker, Object)

    Worker(std::shared_ptr<Chart> const &c = nullptr);

    virtual ~Worker();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual std::string name() const { return m_name_; };

    std::shared_ptr<Chart> const &get_chart() const;

    void set_chart(std::shared_ptr<Chart> const &c);

    std::shared_ptr<model::Model> const &get_model() const;

    virtual void load(data::DataEntityTable const &) { UNIMPLEMENTED; }

    virtual void save(data::DataEntityTable *) const { UNIMPLEMENTED; }

    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m);

    virtual void deploy();

    virtual void pre_process();

    virtual void initialize(Real data_time, Real dt);

    virtual void next_time_step(Real data_time, Real dt) {};

    virtual void phase0(Real data_time, Real dt) { concept::LifeControllable::to_phase(1); };

    virtual void phase1(Real data_time, Real dt) { concept::LifeControllable::to_phase(2); };

    virtual void phase2(Real data_time, Real dt) { concept::LifeControllable::to_phase(3); };

    virtual void phase3(Real data_time, Real dt) { concept::LifeControllable::to_phase(4); };

    virtual void phase4(Real data_time, Real dt) { concept::LifeControllable::to_phase(5); };

    virtual void phase5(Real data_time, Real dt) { concept::LifeControllable::to_phase(6); };

    virtual void phase6(Real data_time, Real dt) { concept::LifeControllable::to_phase(7); };

    virtual void phase7(Real data_time, Real dt) { concept::LifeControllable::to_phase(8); };

    virtual void phase8(Real data_time, Real dt) { concept::LifeControllable::to_phase(9); };

    virtual void phase9(Real data_time, Real dt) { concept::LifeControllable::to_phase(10); };


    virtual unsigned int next_phase(Real data_time, Real dt, unsigned int inc_phase = 1);

    virtual unsigned int max_phase_num() const { return MAX_NUM_OF_PHASE; };


    virtual void sync();

    virtual void finalize(Real data_time, Real dt);

    virtual void post_process();

    virtual void set_physical_boundary_conditions(Real time) {};


    std::shared_ptr<Chart> m_chart_;

protected:

    std::shared_ptr<model::Model> m_model_;
    std::string m_name_;
    static constexpr unsigned int MAX_NUM_OF_PHASE = 9;
};


}}
#endif //SIMPLA_WORKER_H
