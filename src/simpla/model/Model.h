//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/concept/Configurable.h>
#include <simpla/mesh/AttributeView.h>
#include <simpla/mesh/EntityId.h>
#include <simpla/mesh/Mesh.h>
#include "geometry/GeoObject.h"

namespace simpla {
namespace model {
using namespace mesh;
using namespace data;
class GeoObject;

class Model : public concept::Printable, public concept::Configurable {
    typedef Model this_type;

   public:
    enum MODEL_TAG { VACUUM = 1, PLASMA = 1 << 1, CUSTOM = 1 << 20 };
    typedef Mesh::entity_id entity_id;
    Model(Mesh *m);
    virtual ~Model();

    virtual void AddObject(std::string const &name, std::shared_ptr<geometry::GeoObject> const &);

    virtual void RemoveObject(std::string const &key);

    virtual std::ostream &Print(std::ostream &os, int indent) const;

    virtual void Load(std::string const &);

    virtual void Save(std::string const &);

    virtual void Deploy();

    virtual void PreProcess();

    virtual void Initialize(Real data_time = 0, Real dt = 0);

    virtual void NextTimeStep(Real data_time = 0, Real dt = 0);

    virtual void Finalize(Real data_time, Real dt);

    virtual void PostProcess();

    Range<entity_id> const &select(int iform, int tag);

    Range<entity_id> const &select(int iform, std::string const &tag);

    Range<entity_id> const &interface(int iform, const std::string &tag_in, const std::string &tag_out = "VACUUM");

    Range<entity_id> const &interface(int iform, int tag_in, int tag_out);

    Range<entity_id> const &select(int iform, int tag) const { return m_range_cache_.at(iform).at(tag); }

    Range<entity_id> const &interface(int iform, int tag_in, int tag_out = VACUUM) const {
        return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
    }

   private:
    Mesh *m_mesh_;

    DataAttribute<int, VERTEX, 9> m_tags_{m_mesh_, {"name"_ = "tags", "INPUT"}};

    int m_g_obj_count_;

    std::map<std::string, int> m_g_name_map_;

    std::multimap<int, std::shared_ptr<geometry::GeoObject>> m_g_obj_;

    std::map<int, std::map<int, Range<entity_id>>> m_range_cache_;

    std::map<int, std::map<int, std::map<int, Range<entity_id>>>> m_interface_cache_;
};
}
}  // namespace simpla{namespace model{

#endif  // SIMPLA_MODEL_H
