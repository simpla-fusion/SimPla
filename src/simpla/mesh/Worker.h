//
// Created by salmon on 16-10-25.
//

#ifndef SIMPLA_WORKER_H
#define SIMPLA_WORKER_H


namespace simpla { namespace mesh
{
class AttributeBase;

template<typename ...> class Attribute;

template<typename V, typename M, MeshEntityType IFORM> class Patch;

class WorkerBase
{
public:
    typedef toolbox::Object::id_type id_type;

    virtual void move_to(id_type const &mesh_id) {};

    virtual void registerAttribute(std::shared_ptr<AttributeBase> const &attr, std::string const &name)
    {
        m_attributes_[name] = attr;
    };


    template<typename V, typename M, MeshEntityType IFORM>
    void registerAttribute(Attribute<Patch<V, M, IFORM>> &attr, std::string const &name = "")
    {
        typedef Attribute<Patch<V, M, IFORM>>
                f_type;
        static_assert(std::is_base_of<AttributeBase, f_type>::value, "illegal Attribute type");
        registerAttribute(attr.shared_from_this(), name == "" ? attr.name() : name);
    };

    std::map<std::string, std::shared_ptr<AttributeBase> > const &attributes() const { return m_attributes_; };

    std::map<std::string, std::shared_ptr<AttributeBase> > &attributes() { return m_attributes_; };
private:
    std::map<std::string, std::shared_ptr<AttributeBase> > m_attributes_;
};

template<typename ...> class Worker;
}}
#endif //SIMPLA_WORKER_H
