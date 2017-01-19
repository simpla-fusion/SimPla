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
    SP_OBJECT_HEAD(Worker, Object)
   public:
    Worker(Mesh *);

    virtual ~Worker();

    virtual std::ostream &Print(std::ostream &os, int indent = 0) const;

    virtual void Load(data::DataTable const &) { UNIMPLEMENTED; }

    virtual void Save(data::DataTable *) const { UNIMPLEMENTED; }

    virtual Mesh *mesh() { return m_mesh_; };

    virtual Mesh const *mesh() const { return m_mesh_; };

    virtual void Accept(Patch *m);

    virtual void Deploy();

    virtual void PreProcess();

    virtual void Initialize(Real data_time, Real dt);

    virtual void Finalize(Real data_time, Real dt);

    virtual void PostProcess();

    virtual void Destroy();

    virtual void NextTimeStep(Real data_time, Real dt){};

    virtual void Sync();

    virtual void SetPhysicalBoundaryConditions(Real time){};

    Mesh *m_mesh_;
};
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_WORKER_H
