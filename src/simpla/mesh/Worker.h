//
// Created by salmon on 16-11-4.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H

#include <simpla/SIMPLA_config.h>
#include <simpla/design_pattern/Observer.h>
#include <simpla/toolbox/Log.h>
#include <memory>
#include <set>
#include <vector>

#include <simpla/concept/Configurable.h>
#include <simpla/concept/LifeControllable.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>

#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/MeshCommon.h>
#include <simpla/model/Model.h>

#include "Mesh.h"

namespace simpla {
namespace mesh {
struct MeshBlock;
struct DataBlock;

struct Mesh;
struct Patch;

class Worker : public Object,
               public concept::Printable,
               public concept::Serializable,
               public concept::Configurable,
               public concept::LifeControllable {
   public:
    SP_OBJECT_HEAD(Worker, Object)

    Worker();

    virtual ~Worker();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    virtual void load(data::DataTable const &) { UNIMPLEMENTED; }

    virtual void save(data::DataTable *) const { UNIMPLEMENTED; }

    virtual Mesh *mesh() { return m_mesh_.get(); };

    virtual Mesh const *mesh() const { return m_mesh_.get(); };

    virtual void accept(Patch *m);

    virtual void deploy();

    virtual void pre_process();

    virtual void initialize(Real data_time, Real dt);

    virtual void finalize(Real data_time, Real dt);

    virtual void post_process();

    virtual void destroy();

    virtual void next_time_step(Real data_time, Real dt){};

    virtual void sync();

    virtual void set_physical_boundary_conditions(Real time){};

   private:
    std::unique_ptr<Mesh> m_mesh_;
};
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
