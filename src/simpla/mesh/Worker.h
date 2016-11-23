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
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/toolbox/design_pattern/Observer.h>

#include "MeshCommon.h"
#include "Atlas.h"
#include "Attribute.h"

namespace simpla { namespace mesh
{
struct MeshBlock;
struct DataBlock;
struct Attribute;

class Worker :
        public Object,
        public AttributeHolder,
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

    virtual void move_to(std::shared_ptr<mesh::MeshBlock> const &m)=0;

    virtual void initialize(Real data_time)=0;

    virtual void setPhysicalBoundaryConditions(double time)=0;

    virtual void next_time_step(Real data_time, Real dt)=0;

    virtual GeometryBase const *geometry() const =0;

    virtual GeometryBase *geometry()  =0;

    virtual std::shared_ptr<MeshBlock>
    create_mesh_block(int n, index_type const *lo,
                      index_type const *hi,
                      Real const *dx = nullptr,
                      Real const *x0 = nullptr,
                      id_type id = 0)=0;

private:
    std::string m_name_;

};


}}
#endif //SIMPLA_WORKER_H
