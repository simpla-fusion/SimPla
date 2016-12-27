//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <vector>
#include <set>
#include <simpla/toolbox/Log.h>
#include <simpla/design_pattern/Observer.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/concept/LifeControllable.h>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/Attribute.h>
#include "Chart.h"

namespace simpla { namespace mesh
{
struct MeshBlock;
struct DataBlock;

struct Chart;
struct Patch;

class Worker :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable,
        public concept::LifeControllable,
        public AttributeCollection
{
public:
    SP_OBJECT_HEAD(Worker, Object)

    Worker();

    virtual ~Worker();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void load(data::DataTable const &) { UNIMPLEMENTED; }

    virtual void save(data::DataTable *) const { UNIMPLEMENTED; }

//    virtual std::shared_ptr<Chart> clone_mesh() const =0;

    virtual Chart *chart() { return m_chart_.get(); };

    virtual Chart const *chart() const { return m_chart_.get(); };

    virtual void accept(Patch *m);

    virtual void deploy();

    virtual void pre_process();

    virtual void initialize(Real data_time, Real dt);

    virtual void finalize(Real data_time, Real dt);

    virtual void post_process();

    virtual void destroy();

    virtual void next_time_step(Real data_time, Real dt) {};

//    virtual void phase(unsigned int num, Real data_time, Real dt);
//
//    virtual unsigned int next_phase(Real data_time, Real dt, unsigned int inc_phase = 0);


    virtual void sync();

    virtual void set_physical_boundary_conditions(Real time) {};

private:
    std::shared_ptr<Chart> m_chart_;
};


}}
#endif //SIMPLA_WORKER_H
