/** 
 * @file PhysicalDomain.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_PHYSICALDOMAIN_H
#define SIMPLA_PHYSICALDOMAIN_H

#include <memory>
#include "../sp_def.h"
#include "../toolbox/Object.h"
#include "../toolbox/Log.h"
#include "../toolbox/Properties.h"
#include "../toolbox/ConfigParser.h"
#include "../toolbox/IOStream.h"
#include "../mesh/Mesh.h"
#include "../mesh/Atlas.h"
#include "../mesh/TransitionMap.h"
#include "../mesh/Attribute.h"


namespace simpla
{
namespace io { struct IOStream; }
namespace parallel { struct DistributedObject; }

namespace mesh
{
class Block;

class Attribute;
} //namespace get_mesh
} //namespace simpla

namespace simpla { namespace simulation
{


class PhysicalDomain : public toolbox::Object
{
public:
    const mesh::Block *m_mesh_;

    std::shared_ptr<PhysicalDomain> m_next_;

    HAS_PROPERTIES;

    SP_OBJECT_HEAD(PhysicalDomain, toolbox::Object);

    PhysicalDomain();

    PhysicalDomain(const mesh::Block *);

    mesh::Block const *mesh() const { return m_mesh_; }

    virtual  ~PhysicalDomain();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<PhysicalDomain> clone(mesh::Block const &) const;

    virtual bool same_as(mesh::Block const &) const;

    virtual void deploy();

    virtual void teardown();

    virtual void next_step(Real dt) = 0;

    virtual toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const;

    virtual toolbox::IOStream &load(toolbox::IOStream &is) const;

    virtual void sync(mesh::TransitionMap const &, PhysicalDomain const &other);

    std::shared_ptr<PhysicalDomain> &next() { return m_next_; }

    template<typename T, typename ...Args>
    void append_as(Args &&...args)
    {
        append(std::make_shared<T>(mesh(), std::forward<Args>(args)...));
    };

    void append(std::shared_ptr<PhysicalDomain> p_new)
    {
        assert(p_new->mesh()->id() == mesh()->id());

        auto *p = &m_next_;
        while (*p != nullptr) { p = &((*p)->m_next_); } // find tail
        *p = p_new;
    };

    //------------------------------------------------------------------------------------------------------------------
    const mesh::Attribute *attribute(std::string const &s_name) const;

    void add_attribute(mesh::Attribute *attr, std::string const &s_name);

    template<typename TF>
    void global_declare(TF *attr, std::string const &s_name)
    {
        static_assert(std::is_base_of<mesh::Attribute, TF>::value, "illegal Mesh convert");
        add_attribute(dynamic_cast<mesh::Attribute *>(attr), s_name);
    };

//    template<typename TF>
//    std::shared_ptr<TF> create()
//    {
//        auto res = std::make_shared<TF>(m_mesh_);
//        res->deploy();
//        return res;
//    }
//
//    template<typename TF, typename ...Args>
//    std::shared_ptr<TF> create(std::string const &s, Args &&...args)
//    {
//        auto res = std::make_shared<TF>(m_mesh_, std::forward<Args>(args)...);
//        add_attribute(res, s);
//        return res;
//    }

private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_PHYSICALDOMAIN_H
