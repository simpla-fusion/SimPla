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


namespace simpla { namespace io { struct IOStream; }}

namespace simpla { namespace mesh
{
struct MeshAttribute;

class MeshBase;
}}


namespace simpla { namespace task_flow
{


class Worker : public base::Object
{
public:

    SP_OBJECT_HEAD(Worker, base::Object);

    Worker();

    Worker(mesh::MeshBase const &);

    virtual  ~Worker()noexcept;

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; }

    virtual std::shared_ptr<Worker> clone(mesh::MeshBase const &) const
    {
        UNIMPLEMENTED;
        return std::shared_ptr<Worker>(nullptr);
    };

    virtual bool view(mesh::MeshBase const &other)
    {
        m = &other;
        for (auto &item:m_attr_)
        {
            //DO STD
        }
        UNIMPLEMENTED;
    };

    virtual void update_ghost_from(mesh::MeshBase const &other) { };

    virtual bool same_as(mesh::MeshBase const &) const { return false; };

    virtual std::vector<mesh::box_type> refine_boxes() const { return std::vector<mesh::box_type>(); };

    virtual void refine(mesh::MeshBase const &other) { };

    virtual bool coarsen(mesh::MeshBase const &other) { return false; };

    virtual void setup() { };

    virtual void teardown() { };

    virtual io::IOStream &check_point(io::IOStream &os) const;

    virtual io::IOStream &save(io::IOStream &os) const;

    virtual io::IOStream &load(io::IOStream &is) const;

    virtual void view(mesh::MeshBlockId const &) { }

    virtual void next_step(Real dt)
    {
        m_time_ += dt;
        ++m_step_count_;
    }
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
        return std::move(*(m_attr_[s]->template add<TF>(m, std::forward<Args>(args)...)));
    }

private:
    template<typename TF, typename ...Args, size_t ...I>
    TF create_helper_(std::tuple<Args...> const &args, index_sequence<I...>)
    {
        return std::move(create<TF>(std::get<I>(args)...));
    }

public:
    template<typename TF, typename ...Args, size_t ...I>
    TF create(std::tuple<Args...> const &args)
    {
        return std::move(create_helper_<TF>(args, index_sequence_for<Args...>()));
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
