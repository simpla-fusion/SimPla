/** 
 * @file MeshWalker.h
 * @author salmon
 * @date 16-5-23 - 下午2:34
 *  */

#ifndef SIMPLA_MESHWALKER_H
#define SIMPLA_MESHWALKER_H

#include <memory>
#include "../base/Object.h"
#include "../gtl/primitives.h"
#include "../gtl/Log.h"

#include "../mesh/Mesh.h"
#include "../mesh/MeshAttribute.h"


namespace simpla
{
namespace io { struct IOStream; }
namespace parallel { struct DistributedObject; }

namespace mesh
{
class MeshBase;

class MeshAttribute;
} //namespace mesh
} //namespace simpla

namespace simpla
{


class ProblemDomain : public base::Object
{
public:

    SP_OBJECT_HEAD(ProblemDomain, base::Object);

    ProblemDomain();

    ProblemDomain(const mesh::MeshBase *);

    virtual  ~ProblemDomain();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<ProblemDomain> clone(mesh::MeshBase const &) const;

//    virtual bool view(mesh::MeshBase const &other);

//    virtual void view(mesh::MeshBlockId const &);

//    virtual void update_ghost_from(mesh::MeshBase const &other);

    virtual bool same_as(mesh::MeshBase const &) const;

    virtual std::vector<mesh::box_type> refine_boxes() const;

    virtual void refine(mesh::MeshBase const &other);

    virtual bool coarsen(mesh::MeshBase const &other);


    virtual void setup();

    virtual void teardown();

    virtual void sync();

    virtual bool is_ready() const;

    virtual void wait();

    virtual void init() = 0;

    virtual void next_step(Real dt) = 0;


    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;



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

    //------------------------------------------------------------------------------------------------------------------


    Real const &dt() const;

    void dt(Real pdt);

    Real time() const;

    void time(Real t);

    void run(Real stop_time, int num_of_step = 0);


    const mesh::MeshBase *m;

private:

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;

};
}//namespace simpla
#endif //SIMPLA_MESHWALKER_H
