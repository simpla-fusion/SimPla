//
// Created by salmon on 16-11-19.
//

#ifndef SIMPLA_GEOMETRY_H
#define SIMPLA_GEOMETRY_H

#include <simpla/design_pattern/Observer.h>
#include "Attribute.h"
#include "MeshBlock.h"
namespace simpla {
namespace model {
class Model;
}  // namespace model{

namespace mesh {
class Patch;

class DataBlock;

/**
 *  Define:
 *   A bundle is a triple $(E, p, B)$ where $E$, $B$ are sets and $p:E→B$ a map
 *   - $E$ is called the total space
 *   - $B$ is the base space of the bundle
 *   - $p$ is the projection
 *
 */
class Mesh : public concept::Printable, public concept::LifeControllable {
   public:
    SP_OBJECT_BASE(Mesh);
    typedef MeshEntityId entity_id;

    Mesh();

    template <typename... Args>
    Mesh(Args&&... args) : m_mesh_block_(std::make_shared<MeshBlock>(std::forward<Args>(args)...)) {}

    virtual ~Mesh();

    virtual std::ostream& print(std::ostream& os, int indent) const;

    virtual void deploy();

    virtual void pre_process();

    virtual void post_process();

    virtual void initialize(Real data_time = 0, Real dt = 0);

    virtual void finalize(Real data_time = 0, Real dt = 0);

    virtual std::shared_ptr<MeshBlock> const& mesh_block() const {
        ASSERT(m_mesh_block_ != nullptr);
        return m_mesh_block_;
    }

    decltype(auto) dimensions() const { return m_mesh_block_->dimensions(); }

    template <typename... Args>
    Range<MeshEntityId> range(Args&&... args) const {
        if (m_mesh_block_ != nullptr) {
            return m_mesh_block_->range(std::forward<Args>(args)...);
        } else {
            return Range<MeshEntityId>();
        }
    }
    size_type size(int IFORM = VERTEX) const { return m_mesh_block_->number_of_entities(IFORM); }
    template <typename... Args>
    auto hash(Args&&... args) const {
        return m_mesh_block_->hash(std::forward<Args>(args)...);
    }
    template <typename... Args>
    auto pack(Args&&... args) const {
        return m_mesh_block_->pack(std::forward<Args>(args)...);
    }
    point_type dx() const {
        if (m_mesh_block_ != nullptr) {
            return m_mesh_block_->dx();
        } else {
            return point_type{1, 1, 1};
        }
    }
    template <typename... Args>
    decltype(auto) point_global_to_local(Args&&... args) const {
        return m_mesh_block_->point_global_to_local(std::forward<Args>(args)...);
    }
    template <typename... Args>
    decltype(auto) point(Args&&... args) const {
        return m_mesh_block_->point(std::forward<Args>(args)...);
    }

    simpla::model::Model& model() { return *m_model_; }
    simpla::model::Model const& model() const { return *m_model_; }

    virtual void connect(Attribute* attr) { m_attrs_.insert(attr); };
    virtual void disconnect(Attribute* attr) { m_attrs_.erase(attr); }
    virtual void accept(Patch* p);

   protected:
    std::shared_ptr<MeshBlock> m_mesh_block_;
    std::shared_ptr<simpla::model::Model> m_model_;
    std::set<Attribute*> m_attrs_;
};

template <typename...>
class ChartAdapter;

template <typename U>
class ChartAdapter<U> : public Mesh, public U {
    template <typename... Args>
    explicit ChartAdapter(Args&&... args) : U(std::forward<Args>(args)...) {}

    ~ChartAdapter() {}

    virtual std::ostream& print(std::ostream& os, int indent) const {
        U::print(os, indent);
        Mesh::print(os, indent);
    }

    virtual void accept(Patch* p) {
        Mesh::accept(p);
        U::accpt(p);
    };

    virtual void pre_process() {
        Mesh::pre_process();
        U::pre_process();
    };

    virtual void post_process() {
        U::post_process();
        Mesh::post_process();
    };

    virtual void initialize(Real data_time = 0, Real dt = 0) {
        Mesh::initialize(data_time, dt);
        U::initialize(data_time, dt);
    }

    virtual void finalize(Real data_time = 0, Real dt = 0) {
        U::finalize(data_time, dt);
        Mesh::finalize(data_time, dt);
    }
};
}
}  // namespace simpla { namespace mesh

#endif  // SIMPLA_GEOMETRY_H
