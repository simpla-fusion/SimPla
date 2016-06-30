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


    io::IOStream &save_mesh(io::IOStream &os) const;

    io::IOStream &load_mesh(io::IOStream &is);

    io::IOStream &save(io::IOStream &os, int flag = io::SP_NEW) const;

    io::IOStream &load(io::IOStream &is);

    io::IOStream &check_point(io::IOStream &os) const;

    mesh::MeshBlockId add_mesh(std::shared_ptr<mesh::Chart>);

    template<typename TM, typename ...Args>
    std::shared_ptr<TM> add_mesh(Args &&...args)
    {
        auto res = std::make_shared<TM>(std::forward<Args>(args)...);
        add_mesh(std::dynamic_pointer_cast<mesh::Chart>(res));
        return res;
    };

    mesh::Atlas &get_mesh_atlas();

    mesh::Atlas const &get_mesh_atlas() const;

    std::shared_ptr<const mesh::Chart> get_mesh_block(mesh::MeshBlockId id) const;

    std::shared_ptr<mesh::Chart> get_mesh_block(mesh::MeshBlockId id);

    template<typename TM>
    std::shared_ptr<const TM> get_mesh(mesh::MeshBlockId s) const
    {
        return std::dynamic_pointer_cast<const TM>(get_mesh_block(s));
    }

    template<typename TM>
    std::shared_ptr<TM> get_mesh(mesh::MeshBlockId s)
    {
        static_assert(std::is_base_of<mesh::MeshBase, TM>::value, "illegal mesh convert!");
        assert(get_mesh_block(s).get() != nullptr);
        assert(get_mesh_block(s)->is_a<TM>());
        auto res = std::dynamic_pointer_cast<TM>(get_mesh_block(s));
        return std::dynamic_pointer_cast<TM>(get_mesh_block(s));
    }

    std::shared_ptr<ProblemDomain> add_domain(std::shared_ptr<ProblemDomain> pb);


    template<typename TProb, typename ...Args>
    std::shared_ptr<TProb> add_problem_domain(mesh::MeshBlockId id, Args &&...args)
    {
        auto res = std::make_shared<TProb>(
                dynamic_cast<typename TProb::mesh_type const * > (get_mesh_block(id).get()),
                std::forward<Args>(args)...);
        add_domain(res);
        return res;
    };

    template<typename TProb>
    void extend_domain(mesh::MeshBlockId mesh_center, size_type PML_width, std::string prefix = "")
    {


        auto &atlas = get_mesh_atlas();

        auto first = atlas.get_block(mesh_center);

        int m_flag = first->status();

        int od[3];

        int count = 0;

        for (int tag = 1, tag_e = 1 << 6; tag < tag_e; ++tag)
        {
            static constexpr int flag[4] = {0, 1, -1, 2};
            od[0] = flag[tag & 0x3];
            od[1] = flag[(tag >> 2) & 0x3];
            od[2] = flag[(tag >> 4) & 0x3];


            auto second = first->extend(od, PML_width);

            if (second != nullptr)
            {

                auto second_id = atlas.add_block(second);
                second->name(prefix + type_cast<std::string>(count)).deploy();

                atlas.add_adjacency(mesh_center, second_id, mesh::SP_MB_SYNC);
                atlas.add_adjacency(second_id, mesh_center, mesh::SP_MB_SYNC);

                add_domain(std::make_shared<TProb>(static_cast<typename TProb::mesh_type const *>(second.get()),
                                                   od))->deploy();

                ++count;
            }
        }
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

    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};


}}// namespace simpla{namespace simulation


#endif /* CORE_APPLICATION_CONTEXT_H_ */
