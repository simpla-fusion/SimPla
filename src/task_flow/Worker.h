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

namespace mesh
{
class MeshBase;

class MeshAttribute;
} //namespace mesh
} //namespace simpla

namespace simpla { namespace task_flow
{

class Worker : public base::Object
{
public:

    SP_OBJECT_HEAD(Worker, base::Object);

    Worker();

    Worker(mesh::MeshBase const &);

    virtual  ~Worker();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::shared_ptr<Worker> clone(mesh::MeshBase const &) const;

    virtual bool view(mesh::MeshBase const &other);

    virtual void view(mesh::MeshBlockId const &);

    virtual void update_ghost_from(mesh::MeshBase const &other);

    virtual bool same_as(mesh::MeshBase const &) const;

    virtual std::vector<mesh::box_type> refine_boxes() const;

    virtual void refine(mesh::MeshBase const &other);

    virtual bool coarsen(mesh::MeshBase const &other);

    virtual void setup();

    virtual void teardown();

    virtual void next_step(Real dt);

    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;



    //------------------------------------------------------------------------------------------------------------------
    /**
     * factory concept
     */
    static constexpr bool is_factory = true;

    template<typename TF, typename ...Args>
    TF create(std::string const &s, Args &&...args)
    {
        if (m_attr_.find(s) == m_attr_.end())
        {
            m_attr_.emplace(std::make_pair(s, std::make_shared<mesh::MeshAttribute>()));
        }

        std::shared_ptr<TF> res(nullptr);
        try
        {
            res = (m_attr_[s]->template add<TF>(m, std::forward<Args>(args)...));
        }
        catch (...)
        {
            RUNTIME_ERROR << "Can not create attribute [" << s << "]" << std::endl;
        }
        assert(res != nullptr);
        return TF(*res);
    }




    //------------------------------------------------------------------------------------------------------------------



    Real time() const { return m_time_; }

    size_t step_count() const { return m_step_count_; }

    mesh::MeshBase const *m;

private:
    Real m_time_ = 0;
    size_t m_step_count_ = 0;
    std::map<std::string, std::shared_ptr<mesh::MeshAttribute> > m_attr_;
};
}}//namespace simpla{namespace mesh{
#endif //SIMPLA_MESHWALKER_H
