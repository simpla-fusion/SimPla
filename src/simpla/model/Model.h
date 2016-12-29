//
// Created by salmon on 16-11-27.
//

#ifndef SIMPLA_MODEL_H
#define SIMPLA_MODEL_H

#include <simpla/mesh/Chart.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/concept/Configurable.h>
#include <simpla/geometry/GeoObject.h>

namespace simpla { namespace model
{
using namespace mesh;

class GeoObject;

class Model :
        public Object,
        public concept::Printable,
        public concept::Configurable,
        public concept::LifeControllable,
        public mesh::AttributeCollection
{
public:

    enum MODEL_TAG { VACUUME = 1, PLASMA = 1 << 1, CUSTOM = 1 << 20 };

    SP_OBJECT_HEAD(Model, Object)

    Model();

    virtual ~Model();

    virtual void add_object(std::string const &name, std::shared_ptr<geometry::GeoObject> const &);

    virtual void remove_object(std::string const &key);

    virtual std::ostream &print(std::ostream &os, int indent) const;

    virtual void load(std::string const &);

    virtual void save(std::string const &);

    virtual void deploy();

    virtual void pre_process();

    virtual void initialize(Real data_time = 0, Real dt = 0);

    virtual void next_time_step(Real data_time = 0, Real dt = 0);

    virtual void finalize(Real data_time, Real dt);

    virtual void post_process();

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, int tag);

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, std::string const &tag);

    virtual mesh::EntityIdRange const &
    interface(MeshEntityType iform, const std::string &tag_in, const std::string &tag_out = "VACUUME");

    mesh::EntityIdRange const &interface(MeshEntityType iform, int tag_in, int tag_out);

    virtual mesh::EntityIdRange const &
    select(MeshEntityType iform, int tag) const
    {
        return m_range_cache_.at(iform).at(tag);
    }

    virtual mesh::EntityIdRange const &
    interface(MeshEntityType iform, int tag_in, int tag_out = VACUUME) const
    {
        return m_interface_cache_.at(iform).at(tag_in).at(tag_out);
    }

private:

    Variable<int, VERTEX, 9> m_tags_{"name=tags;INPUT"};

    std::shared_ptr<Chart> m_chart_ = nullptr;

    int m_g_obj_count_;

    std::map<std::string, int> m_g_name_map_;

    std::multimap<int, std::shared_ptr<geometry::GeoObject>> m_g_obj_;

    std::map<id_type, std::map<int, EntityIdRange>> m_range_cache_;

    std::map<id_type, std::map<int, std::map<int, EntityIdRange>>> m_interface_cache_;


};
}}//namespace simpla{namespace model{

#endif //SIMPLA_MODEL_H
