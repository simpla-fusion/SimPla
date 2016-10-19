/** 
 * @file PhysicalDomain.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_PHYSICALDOMAIN_H
#define SIMPLA_PHYSICALDOMAIN_H

#include <memory>
#include "SIMPLA_config.h"
#include "../toolbox/Object.h"
#include "../toolbox/Log.h"
#include "../toolbox/Properties.h"
#include "../toolbox/ConfigParser.h"
#include "../toolbox/IOStream.h"
#include "../mesh/Mesh.h"
#include "../mesh/Atlas.h"
#include "../mesh/TransitionMap.h"
#include "../mesh/Attribute.h"


namespace simpla { namespace simulation
{

/**
 *
 */
class DomainBase : public toolbox::Object
{
public:
    std::shared_ptr<mesh::MeshBase> m;

    std::shared_ptr<DomainBase> m_sub_level_;

    std::shared_ptr<DomainBase> m_next_;

    HAS_PROPERTIES;

    SP_OBJECT_HEAD(DomainBase, toolbox::Object);

    DomainBase();

    DomainBase(std::shared_ptr<const mesh::MeshBase>);

    virtual std::shared_ptr<mesh::MeshBase> mesh() const { return m; };

    virtual ~DomainBase();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<DomainBase> clone() const =0;

    virtual std::shared_ptr<DomainBase> refine(index_box_type const &b, int n = 1, int flag = 0) const;

    virtual void deploy();

    virtual void teardown();

    virtual void next_step(Real dt) = 0;

    virtual toolbox::IOStream &save(toolbox::IOStream &os, int flag = toolbox::SP_NEW) const;

    virtual toolbox::IOStream &load(toolbox::IOStream &is) const;

    virtual void sync(mesh::TransitionMapBase const &, DomainBase const &other);

    std::shared_ptr<DomainBase> &next() { return m_next_; }

    template<typename T, typename ...Args>
    void append_as(Args &&...args)
    {
        append(std::make_shared<T>(mesh(), std::forward<Args>(args)...));
    };

    void append(std::shared_ptr<DomainBase> p_new)
    {
        assert(p_new->mesh()->id() == mesh()->id());

        auto *p = &m_next_;
        while (*p != nullptr) { p = &((*p)->m_next_); } // find tail
        *p = p_new;
    };

    //------------------------------------------------------------------------------------------------------------------
    const mesh::AttributeBase *attribute(std::string const &s_name) const;

    void add_attribute(mesh::AttributeBase *attr, std::string const &s_name);

    template<typename TF>
    void global_declare(TF *attr, std::string const &s_name)
    {
        static_assert(std::is_base_of<mesh::AttributeBase, TF>::value, "illegal type conversion");
        add_attribute(dynamic_cast<mesh::AttributeBase *>(attr), s_name);
    };


private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_PHYSICALDOMAIN_H
