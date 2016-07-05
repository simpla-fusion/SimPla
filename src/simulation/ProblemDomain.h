/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../sp_def.h"
#include "../base/Object.h"
#include "../gtl/Log.h"
#include "../gtl/Properties.h"
#include "../gtl/ConfigParser.h"
#include "../mesh/Mesh.h"
#include "../mesh/MeshAtlas.h"
#include "../mesh/MeshAttribute.h"


namespace simpla
{
namespace io { struct IOStream; }
namespace parallel { struct DistributedObject; }

namespace mesh
{
class MeshBase;

class MeshAttribute;
} //namespace get_mesh
} //namespace simpla

namespace simpla { namespace simulation
{


class ProblemDomain : public base::Object
{
public:
    const mesh::MeshBase *m_mesh_;

    std::shared_ptr<ProblemDomain> m_next_;

    HAS_PROPERTIES;

    SP_OBJECT_HEAD(ProblemDomain, base::Object);

    ProblemDomain();

    ProblemDomain(const mesh::MeshBase *);

    mesh::MeshBase const *mesh() const { return m_mesh_; }

    virtual  ~ProblemDomain();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<ProblemDomain> clone(mesh::MeshBase const &) const;

    virtual bool same_as(mesh::MeshBase const &) const;

    virtual void deploy();

    virtual void teardown();

    virtual void next_step(Real dt) = 0;

    virtual io::IOStream &save(io::IOStream &os, int flag = io::SP_NEW) const;

    virtual io::IOStream &load(io::IOStream &is) const;

    virtual void sync(mesh::TransitionMap const &, ProblemDomain const &other);

    std::shared_ptr<ProblemDomain> &next() { return m_next_; }

    template<typename T, typename ...Args>
    void append_as(Args &&...args)
    {
        append(std::make_shared<T>(mesh(), std::forward<Args>(args)...));
    };

    void append(std::shared_ptr<ProblemDomain> p_new)
    {
        assert(p_new->mesh()->id() == mesh()->id());

        auto *p = &m_next_;
        while (*p != nullptr) { p = &((*p)->m_next_); } // find tail
        *p = p_new;
    };

    //------------------------------------------------------------------------------------------------------------------
    const mesh::MeshAttribute *attribute(std::string const &s_name) const;

    void add_attribute(mesh::MeshAttribute *attr, std::string const &s_name);

    template<typename TF>
    void declare_global(TF *attr, std::string const &s_name)
    {
        static_assert(std::is_base_of<mesh::MeshAttribute, TF>::value, "illegal Mesh convert");
        add_attribute(dynamic_cast<mesh::MeshAttribute *>(attr), s_name);
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

#endif //SIMPLA_MESHWALKER_H
