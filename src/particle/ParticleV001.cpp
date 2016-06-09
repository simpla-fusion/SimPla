//
// Created by salmon on 16-6-8.
//
#include <cstddef>
#include "../sp/SmallObjPool.h"
#include "../mesh/MeshCommon.h"
#include "ParticleCommon.h"
#include "ParticleV001.h"

namespace simpla { namespace particle
{

void _impl::load_dataset(mesh::MeshEntityRange const &r0, data_model::DataSet const &ds,
                         sp::spPagePool *pool,
                         parallel::concurrent_hash_map<typename mesh::MeshEntityId, sp::spPage *> *data)
{
    typedef parallel::concurrent_hash_map<typename mesh::MeshEntityId, sp::spPage *> container_type;
    std::mutex m_pool_mutex_;
    sp::spPage *pg = 0x0;
    size_t num_of_element = ds.memory_space.num_of_elements();

    if (num_of_element == 0) { return; }

    void const *src = ds.data.get();

    size_t size_in_byte = ds.data_type.size_in_byte();


    parallel::parallel_for(
            r0, [&](mesh::MeshEntityRange const &r)
            {

                for (auto const &key:r)
                {
                    sp::spPage *pg = 0x0;
                    while (num_of_element > 0)
                    {
                        {
                            std::unique_lock<std::mutex> pool_lock(m_pool_mutex_);
                            sp::spPage *t = sp::spPageCreate(pool);
                            sp::spPushFront(&pg, &t);
                        }

                        sp::status_tag_type tag = 0x1;
                        void *dest = pg->data;

                        for (; tag != 0x0 && num_of_element > 0;
                               --num_of_element, src += size_in_byte)
                        {
                            if (reinterpret_cast<point_head *>(src)->_cell != key)
                            {
                                memcpy(dest, src, size_in_byte);
                                tag <<= 1;
                                dest += size_in_byte;
                                pg->tag |= tag;
                            }
                        }
                    }

                    typename container_type::accessor accessor;
                    if (data->insert(accessor, key)) { sp::spPushFront(&(accessor->second), &pg); }
                }
            }
    );
}


}


}
}//namespace simpla { namespace simpla