/**
 * @file model.h
 *
 *  created on: 2013-12-15
 *      Author: salmon
 */

#ifndef MODEL_H_
#define MODEL_H_

#include <stddef.h>
#include <algorithm>
#include <bitset>
#include <limits>
#include <map>
#include <string>
#include <memory>

#include "model_base.h"

#include "../sp_def.h"
#include "../gtl/type_traits.h"
#include "../gtl/utilities/utilities.h"
#include "../manifold/topology/block.h"
#include "../manifold/obsolete/mesh_block.h.h"

namespace simpla
{

/**
 *  @defgroup  model model
 *  @brief BaseManifold modeling
 */

/**
 *  @ingroup model
 *   @brief model
 */

class Model
{

public:


    typedef typename Block::id_type id_type;

    typedef std::uint64_t tag_type;

    static constexpr int MAX_NUM_OF_MEDIA_TYPE = std::numeric_limits<tag_type>::digits;

    static constexpr tag_type null_material = 0UL;

    std::shared_ptr<tag_type> m_data_;

    std::map<std::string, tag_type> registered_material_;

    int max_material_;

    Block const &m_geo_;
public:

    enum
    {
        NONE = 0,
        VACUUM = 1UL << 1,
        PLASMA = 1UL << 2,
        CORE = 1UL << 3,
        BOUNDARY = 1UL << 4,
        LIMITER = 1UL << 5,
        // @NOTE: add materials for different physical area or media
                CUSTOM = 1UL << 20
    };

    Model(Block const &geo)
            : max_material_(CUSTOM << 1), m_geo_(geo)
    {
        registered_material_.emplace("NONE", null_material);

        registered_material_.emplace("Vacuum", (VACUUM));
        registered_material_.emplace("Plasma", (PLASMA));
        registered_material_.emplace("Core", (CORE));
        registered_material_.emplace("Boundary", (BOUNDARY));
        registered_material_.emplace("Limiter", (LIMITER));

    }

    ~Model()
    {
    }

    bool empty() const
    {
        return m_data_ == nullptr;
    }

    int register_material(std::string const &name)
    {
        int res;
        if (registered_material_.find(name) != registered_material_.end())
        {
            res = registered_material_[name];
        }
        else if (max_material_ < MAX_NUM_OF_MEDIA_TYPE)
        {
            max_material_ = max_material_ << 1;

            res = (max_material_);

        }
        else
        {
            THROW_EXCEPTION_RUNTIME_ERROR("Too much media Type");
        }
        return res;
    }

    int get_material(std::string const &name) const
    {

        if (name == "" || name == "NONE")
        {
            return null_material;
        }
        int res;

        try
        {
            res = registered_material_.at(name);

        } catch (...)
        {
            THROW_EXCEPTION_RUNTIME_ERROR("Unknown material name : " + name);
        }
        return std::move(res);
    }


    tag_type &at(id_type s)
    {
        return m_data_.get()[m_geo_.hash(s)];
    }

    tag_type const &at(id_type s) const
    {
        return m_data_.get()[m_geo_.hash(s)];
    }

    tag_type operator[](id_type s) const
    {
        return at(s);
    }

    void clear()
    {
        m_data_ = nullptr;
    }


    void set(id_type const &s, tag_type const &tag)
    {

        id_type v[Block::MAX_NUM_OF_VERTEX];
        int num = m_geo_.get_adjoin_vertrics(s, v);
        for (int i = 0; i < num; ++i)
        {
            at(v[i]) |= tag;
        }

    }

    void unset(id_type const &s, tag_type const &tag)
    {

        id_type v[Block::MAX_NUM_OF_VERTEX];
        int num = m_geo_.get_adjoin_vertrics(s, v);
        for (int i = 0; i < num; ++i)
        {
            at(v[i]) &= ~tag;
        }

    }

    tag_type get(id_type const &s) const
    {

        id_type v[Block::MAX_NUM_OF_VERTEX];
        int num = m_geo_.get_adjoin_vertrics(s, v);

        tag_type res = null_material;

        for (int i = 0; i < num; ++i)
        {
            res |= at(v[i]);
        }

        return res;

    }


    template<typename TR>
    void erase(TR const &r)
    {
        set(r, null_material);
    }

    template<typename TR>
    void set(TR const &r, tag_type const &tag)
    {
        for (auto s : r)
        {
            set(s, tag);
        }
    }

    template<typename TR>
    void unset(TR const &r, tag_type const &tag)
    {
        for (auto s : r)
        {
            at(s) &= ~tag;
        }
    }

    template<typename TR>
    tag_type get(TR const &r, tag_type const &tag) const
    {
        tag_type res = null_material;

        for (auto s : r)
        {
            res |= get(s);
        }
        return res;
    }

};


}
// namespace simpla

#endif /* MODEL_H_ */
