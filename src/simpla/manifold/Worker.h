//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <memory>
#include <vector>
#include <set>
#include <simpla/data/DataBase.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/design_pattern/Observer.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/manifold/Atlas.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/model/Model.h>
#include "CoordinateFrame.h"
#include "Chart.h"

namespace simpla { namespace mesh
{
struct MeshBlock;
struct DataBlock;

struct ChartBase;
struct CoordinateFrame;

class Worker :
        public Object,
        public concept::Printable,
        public concept::Serializable
{
public:
    SP_OBJECT_HEAD(Worker, Object)

    Worker();

    ~Worker();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual std::string name() const { return m_name_; };

    virtual void load(data::DataBase const &) { UNIMPLEMENTED; }

    virtual void save(data::DataBase *) const { UNIMPLEMENTED; }

    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m);

    virtual void deploy() {};

    virtual void initialize(Real data_time);

    virtual void set_physical_boundary_conditions(Real time) {};

    virtual void next_time_step(Real data_time, Real dt) {};

    virtual ChartBase *chart()=0;

    virtual ChartBase const *chart() const =0;

    virtual model::Model *model()=0;

    virtual model::Model const *model() const =0;


private:


    std::string m_name_;

};


}}
#endif //SIMPLA_WORKER_H
