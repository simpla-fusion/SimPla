//
// Created by salmon on 16-10-25.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H


#include "Context.h"

namespace simpla { namespace simulation
{


template<typename V, typename M, mesh::MeshEntityType IFORM> class Patch;

class WorkerBase
{
public:
    typedef toolbox::Object::id_type id_type;

    virtual void move_to(id_type const &mesh_id) {};

    virtual void registerAttribute(std::shared_ptr<mesh::AttributeBase> const &attr, std::string const &name)
    {
        m_attributes_[name] = attr;
    };

    virtual void registerAttributeTo(ContextBase *)=0;

    template<typename V, typename M, mesh::MeshEntityType IFORM>
    void registerAttribute(mesh::Attribute<Patch<V, M, IFORM>> &attr, std::string const &name = "")
    {
        typedef mesh::Attribute<Patch<V, M, IFORM>> f_type;
        static_assert(std::is_base_of<mesh::AttributeBase, f_type>::value, "illegal Attribute type");
        registerAttribute(attr.shared_from_this(), name == "" ? attr.name() : name);
    };

    std::map<std::string, std::shared_ptr<mesh::AttributeBase> > const &attributes() const { return m_attributes_; };

    std::map<std::string, std::shared_ptr<mesh::AttributeBase> > &attributes() { return m_attributes_; };
private:
    std::map<std::string, std::shared_ptr<mesh::AttributeBase> > m_attributes_;
};

template<typename ...> class Worker;
}}
#endif //SIMPLA_WORKER_H
