/** 
 * @file PhysicalDomain.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_PHYSICALDOMAIN_H
#define SIMPLA_PHYSICALDOMAIN_H

#include <simpla/SIMPLA_config.h>

#include <memory>

#include <simpla/toolbox/Object.h>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/Properties.h>
#include <simpla/toolbox/ConfigParser.h>
#include <simpla/toolbox/IOStream.h>


#include "Mesh.h"
#include "Atlas.h"
#include "TransitionMap.h"
#include "Attribute.h"


namespace simpla { namespace mesh
{

/**
 * Domain: define a physical domain
 *  - HAS-A mesh
 *  - HAS-N attributes
 */
class DomainBase : public toolbox::Object
{
public:

    HAS_PROPERTIES;

    SP_OBJECT_HEAD(DomainBase, toolbox::Object);

    DomainBase();

    DomainBase(std::shared_ptr<Atlas>);

    virtual std::shared_ptr<Atlas> mesh() const;

    virtual ~DomainBase();

    using toolbox::Object::is_a;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<DomainBase> clone() const =0;

    virtual std::shared_ptr<DomainBase> refine(index_box_type const &b, int n = 1, int flag = 0) const;

    virtual void deploy();

    virtual void teardown();

    virtual void next_step(Real dt) = 0;


    virtual void sync(TransitionMapBase const &, DomainBase const &other);


    //------------------------------------------------------------------------------------------------------------------
    std::shared_ptr<AttributeBase> attribute(uuid id);

    void add_attribute(PatchBase *attr, std::string const &s_name);

    template<typename TF>
    void global_declare(TF *attr, std::string const &s_name)
    {
        static_assert(std::is_base_of<PatchBase, TF>::value, "illegal type conversion");
        add_attribute(dynamic_cast<PatchBase *>(attr), s_name);
    };

    void move_to(uuid);

private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_PHYSICALDOMAIN_H
