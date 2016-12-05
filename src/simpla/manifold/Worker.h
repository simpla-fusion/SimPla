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
#include <simpla/concept/Deployable.h>

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
        public concept::Deployable
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

    virtual void initialize(Real data_time = 0);

    virtual void next_time_step(Real data_time, Real dt) {};

    virtual void finalize(Real data_time = 0);

    virtual void post_process();

    virtual void set_physical_boundary_conditions(Real time) {};


    std::shared_ptr<Chart> m_chart_;

protected:

    std::shared_ptr<model::Model> m_model_;
    std::string m_name_;
};


}}
#endif //SIMPLA_WORKER_H
