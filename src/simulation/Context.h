/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <memory>
#include <list>
#include <map>
#include "../sp_def.h"
#include "../mesh/MeshEntity.h"
#include "../mesh/MeshAttribute.h"
#include "../mesh/MeshAtlas.h"
#include "../io/IOStream.h"
#include "ProblemDomain.h"


namespace simpla { namespace simulation
{


class ProblemDomain;

class Context
{
private:
    typedef Context this_type;
public:

    Context();

    ~Context();

    int m_refine_ratio = 2;

    void setup();

    void teardown();


    std::ostream &print(std::ostream &os, int indent = 1) const;

    io::IOStream &check_point(io::IOStream &os) const;

    io::IOStream &save(io::IOStream &os) const;

    io::IOStream &load(io::IOStream &is);


    void add_mesh(std::shared_ptr<mesh::Chart>, int level = 0);

    template<typename TM, typename ...Args>
    std::shared_ptr<TM> add_mesh(int level = 0, Args &&...args)
    {
        auto res = std::make_shared<TM>(std::forward<Args>(args)...);
        add_mesh(std::dynamic_pointer_cast<mesh::Chart>(res), level);
        return res;
    };

    std::shared_ptr<mesh::Chart> get_mesh_chart(mesh::MeshBlockId id, int level = 0) const;

    template<typename TM, typename ...Args>
    std::shared_ptr<const TM> get_mesh(Args &&...args) const
    {
        return std::dynamic_pointer_cast<const TM>(std::forward<Args>(args)...);
    }


    std::shared_ptr<ProblemDomain> add_domain(std::shared_ptr<ProblemDomain> pb, int level = 0);


    template<typename TProb, typename TM>
    std::shared_ptr<TProb> add_problem_domain(std::shared_ptr<TM> m, int level = 0)
    {
        auto res = std::make_shared<TProb>(m.get());
        add_domain(res, level);
        return res;
    };

    template<typename TProb>
    std::shared_ptr<TProb> add_problem_domain(mesh::MeshBlockId id, int level = 0)
    {
        return add_problem_domain<TProb>(get_mesh_chart(id, level), level);
    };

    std::shared_ptr<ProblemDomain> get_domain(mesh::MeshBlockId id) const;

    template<typename TProb, typename ...Args>
    std::shared_ptr<TProb> get_problem_domain(Args &&...args) const
    {
        return std::dynamic_pointer_cast<TProb>(get_domain(std::forward<Args>(args)...));

    }


    void update(int level = 0, int flag = mesh::SP_MB_SYNC);

    void run(Real dt, int level = 0);

    //------------------------------------------------------------------------------------------------------------------
    Real time() const { return m_time_; }

    void time(Real t) { m_time_ = t; };

    void next_time_step(Real dt) { m_time_ += dt; };

private:
    Real m_time_;
private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


}}// namespace simpla{namespace simulation


#endif /* CORE_APPLICATION_CONTEXT_H_ */
