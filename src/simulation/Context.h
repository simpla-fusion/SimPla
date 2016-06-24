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
#include "../gtl/primitives.h"
#include "../mesh/MeshEntity.h"
#include "../mesh/MeshAttribute.h"
#include "../mesh/MeshAtlas.h"
#include "../io/IOStream.h"

#include "../manifold/Manifold.h"

#include "ProblemDomain.h"

namespace simpla
{
namespace mesh { struct MeshBase; }

class Context
{
private:
    typedef Context this_type;


    typedef typename mesh::MeshBlockId block_id;

public:
    static constexpr int MAX_MESH_LEVEL = 10;

    manifold::Atlas m_atlas_;

    std::multimap<mesh::MeshBlockId, std::shared_ptr<ProblemDomain>> m_domains_;

    int m_refine_ratio = 2;

    void setup();

    void teardown();

    io::IOStream &check_point(io::IOStream &os) const;

    io::IOStream &save(io::IOStream &os) const;

    io::IOStream &load(io::IOStream &is);

    void update(int level, int flag = 0);

    void run(Real dt, int level = 1);


//    template<typename TF>
//    std::shared_ptr<TF> get_attribute(std::string const &s_name)
//    {
//        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");
//
//        auto it = m_attributes_.find(s_name);
//
//        if (it == m_attributes_.end())
//        {
//            return create_attribute<TF>(s_name);
//        }
//        else if (it->second.lock()->is_a(typeid(TF)))
//        {
//            return std::dynamic_pointer_cast<TF>(it->second.lock());
//        }
//        else
//        {
//            return nullptr;
//        }
//
//    }
//
//    template<typename TF>
//    std::shared_ptr<TF> create_attribute(std::string const &s_name = "")
//    {
//        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");
//
//        auto res = std::make_shared<TF>(*this);
//
//        if (s_name != "") { enroll(s_name, std::dynamic_pointer_cast<Attribute>(res)); }
//
//        return res;
//    }
//
//    template<typename TF>
//    std::shared_ptr<TF> create_attribute() const
//    {
//        return std::make_shared<TF>(*this);
//    }
//
//    template<typename TF>
//    void enroll(std::string const &name, std::shared_ptr<TF> p)
//    {
//        static_assert(std::is_base_of<Attribute, TF>::value, "TF is not a Attribute");
//
//        m_attributes_.insert(std::make_pair(name, std::dynamic_pointer_cast<Attribute>(p)));
//    };


    template<typename TSolver, typename TM, typename ...Args>
    std::shared_ptr<TSolver> register_solver(TM const &m, Args &&...args)
    {
        static_assert(std::is_base_of<ProblemDomain, TSolver>::value, "TSovler is not derived from ProblemDomain.");
        auto res = std::make_shared<TSolver>(m, std::forward<Args>(args)...);
        m_workers_.emplace(std::make_pair(m.uuid(), std::dynamic_pointer_cast<ProblemDomain>(res)));
        return res;
    }

    template<typename TSolver>
    std::shared_ptr<TSolver> get_worker(mesh::MeshBlockId const &w_id) const
    {
        assert(m_workers_.at(w_id)->template is_a<TSolver>());
        return std::dynamic_pointer_cast<TSolver>(m_workers_.at(w_id));
    }

    std::map<mesh::MeshBlockId, std::shared_ptr<ProblemDomain> > m_workers_;


};


}// namespace simpla

#endif /* CORE_APPLICATION_CONTEXT_H_ */
