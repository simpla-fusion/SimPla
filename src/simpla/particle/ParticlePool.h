//
// Created by salmon on 17-8-7.
//

#ifndef SIMPLA_BUCKETPOOL_H
#define SIMPLA_BUCKETPOOL_H

#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "simpla/SIMPLA_config.h"
#include "simpla/utilities/type_traits.h"
namespace simpla {

struct ParticlePool {
    ParticlePool();
    ~ParticlePool();

    ParticlePool(ParticlePool const &) = delete;
    ParticlePool(ParticlePool &&) = delete;
    ParticlePool &operator=(ParticlePool const &) = delete;
    ParticlePool &operator=(ParticlePool &&) = delete;

    template <typename U>
    void AddDOF(std::string const &s_name, U *ptr = nullptr) {
        AddDOF(s_name, typeid(U), sizeof(U), ptr);
    }
    void AddDOF(std::string const &s_name, std::type_info const &t_info, size_t t_size, void *ptr);
    void Initialize();

    struct Bucket {
        Bucket *next = nullptr;
        int m_tail_ = 0;
        int *m_idx_ = nullptr;
        void **m_data_ = nullptr;
    };
    Bucket *GetBucket(id_type s);
    Bucket const *GetBucket(id_type s) const;

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

//    private:
//    template <int... N, typename... U>
//    auto GetBucketHelp(std::index_sequence<N...> _, Bucket *b) const {
//        return b == nullptr ? std::make_pair(-1, std::tuple<int *, U *...>{})
//                            : std::make_pair(
//                        b->m_tail_,
//                        std::make_tuple(
//                                b->m_idx_, reinterpret_cast<traits::type_list_N_t<N, U...> *>(b->m_data_[N])...));
//    }
//    template <int... N, typename... U>
//    auto GetBucketHelp(std::index_sequence<N...> _, Bucket const *b) const {
//        return b == nullptr
//               ? std::make_pair(-1, std::tuple<int *, U const *...>{})
//               : std::make_pair(b->m_tail_,
//                                std::make_tuple(b->m_idx_, reinterpret_cast<traits::type_list_N_t<N, U...> const *>(
//                                        b->m_data_[N])...));
//    }
//
//    public:
//    /**
//     *
//     * @tparam U
//     * @param s
//     * @return  std::pair<int ,       -- tail/ number of entity
//     *     std::tuple<int *, U*...>>  -- data tuple
//     *
//     */
//    template <typename... U>
//    auto GetBucket(id_type s) {
//        return GetBucketHelp(std::index_sequence_for<U...>(), GetRawBucket(s));
//    };
//    template <typename... U>
//    auto GetBucket(id_type s) const {
//        return GetBucketHelp(std::index_sequence_for<U...>(), GetRawBucket(s));
//    };
}
#endif  // SIMPLA_BUCKETPOOL_H
