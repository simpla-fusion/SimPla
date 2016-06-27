/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../base/Object.h"
#include "../sp_def.h"
#include "../gtl/Log.h"

#include "../mesh/Mesh.h"
#include "../mesh/MeshAtlas.h"
#include "../mesh/MeshAttribute.h"
#include "../gtl/ConfigParser.h"


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

    SP_OBJECT_HEAD(ProblemDomain, base::Object);

    ProblemDomain();

    ProblemDomain(const mesh::MeshBase *);

    mesh::MeshBlockId mesh_id() const { return m->uuid(); }

    virtual  ~ProblemDomain();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<ProblemDomain> clone(mesh::MeshBase const &) const;

    virtual bool same_as(mesh::MeshBase const &) const;

    virtual void setup(ConfigParser const &dict);

    virtual void teardown();

    virtual void init(ConfigParser const &options) = 0;

    virtual void next_step(Real dt) = 0;

    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;

    virtual void sync(mesh::TransitionMap const &, ProblemDomain const &other) = 0;



    //------------------------------------------------------------------------------------------------------------------
    /**
     * factory concept
     */
    static constexpr bool is_factory = true;

    std::shared_ptr<mesh::MeshAttribute> attribute(std::string const &s_name);

    std::shared_ptr<mesh::MeshAttribute const> attribute(std::string const &s_name) const;

    template<typename TF, typename ...Args>
    std::shared_ptr<TF> create(std::string const &s, Args &&...args)
    {
        return (attribute(s)->template add<TF>(m, std::forward<Args>(args)...));
    }


    const mesh::MeshBase *m;

private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}}//namespace simpla { namespace simulation

#endif //SIMPLA_MESHWALKER_H
