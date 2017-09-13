//
// Created by salmon on 17-9-8.
//

#ifndef SIMPLA_MESHCOMMON_H
#define SIMPLA_MESHCOMMON_H

namespace simpla {
namespace mesh {

#define SP_MESH_POLICY_HEAD(_NAME_)                                \
   private:                                                        \
    typedef THost host_type;                                       \
    typedef _NAME_<THost> this_type;                               \
    THost *m_host_;                                                \
                                                                   \
   public:                                                         \
    _NAME_(THost *h) : m_host_(h) {}                               \
    virtual ~_NAME_() = default;                                   \
    _NAME_(_NAME_ const &other) = delete;                          \
    _NAME_(_NAME_ &&other) = delete;                               \
    _NAME_ &operator=(_NAME_ const &other) = delete;               \
    _NAME_ &operator=(_NAME_ &&other) = delete;                    \
    static std::string RegisterName() { return __STRING(_NAME_); } \
    std::shared_ptr<data::DataNode> Serialize() const;             \
    void Deserialize(std::shared_ptr<data::DataNode> const &cfg);

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_MESHCOMMON_H
