//
// Created by salmon on 17-7-18.
//

#ifndef SIMPLA_POLICIESCOMMON_H
#define SIMPLA_POLICIESCOMMON_H

#define SP_ENGINE_POLICY_HEAD(_NAME_)                              \
   private:                                                        \
    typedef THost host_type;                                       \
    typedef _NAME_<THost> this_type;                               \
                                                                   \
   public:                                                         \
    host_type *m_host_ = nullptr;                                  \
    _NAME_(host_type *h) noexcept : m_host_(h) {}                  \
    virtual ~_NAME_() = default;                                   \
    _NAME_(_NAME_ const &other) = delete;                          \
    _NAME_(_NAME_ &&other) = delete;                               \
    _NAME_ &operator=(_NAME_ const &other) = delete;               \
    _NAME_ &operator=(_NAME_ &&other) = delete;                    \
    static std::string RegisterName() { return __STRING(_NAME_); } \
    std::shared_ptr<data::DataNode> Serialize() const;             \
    void Deserialize(std::shared_ptr<const data::DataNode> cfg);

#define DEFINE_INVOKE_HELPER(_FUN_NAME_)                                                                           \
    CHECK_MEMBER_FUNCTION(has_mem_fun_##_FUN_NAME_, _FUN_NAME_)                                                    \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::true_type const &has_function, this_type *self, Args &&... args) {               \
        self->_FUN_NAME_(std::forward<Args>(args)...);                                                             \
        return 1;                                                                                                  \
    }                                                                                                              \
    template <typename this_type, typename... Args>                                                                \
    int _invoke_##_FUN_NAME_(std::false_type const &has_not_function, this_type *self, Args &&... args) {          \
        return 0;                                                                                                  \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type const *self, Args &&... args) {                                         \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type> const, void, Args...>(),               \
                                    dynamic_cast<_T0<this_type> const *>(self), std::forward<Args>(args)...);      \
    }                                                                                                              \
    template <template <typename> class _T0, typename this_type, typename... Args>                                 \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _invoke_##_FUN_NAME_(has_mem_fun_##_FUN_NAME_<_T0<this_type>, void, Args...>(),                     \
                                    dynamic_cast<_T0<this_type> *>(self), std::forward<Args>(args)...);            \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_##_FUN_NAME_(this_type *self, Args &&... args) {                                               \
        return _try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) +                                  \
               _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                      \
    }                                                                                                              \
    template <template <typename> class _T0, template <typename> class _T1, template <typename> class... _TOthers, \
              typename this_type, typename... Args>                                                                \
    int _try_invoke_once_##_FUN_NAME_(this_type *self, Args &&... args) {                                          \
        if (_try_invoke_##_FUN_NAME_<_T0>(self, std::forward<Args>(args)...) == 0) {                               \
            return _try_invoke_##_FUN_NAME_<_T1, _TOthers...>(self, std::forward<Args>(args)...);                  \
        } else {                                                                                                   \
            return 1;                                                                                              \
        }                                                                                                          \
    }
namespace simpla {
namespace traits {

DEFINE_INVOKE_HELPER(InitialCondition)
DEFINE_INVOKE_HELPER(BoundaryCondition)
DEFINE_INVOKE_HELPER(Advance)
DEFINE_INVOKE_HELPER(TagRefinementCells)

DEFINE_INVOKE_HELPER(Deserialize)
DEFINE_INVOKE_HELPER(Serialize)

}  // namespace traits
}  // namespace simpla
#endif  // SIMPLA_POLICIESCOMMON_H
