#include "simpla/core/domain.hpp"
#include "simpla/utils/logger.hpp"

namespace simpla {

class ContextBase {
   protected:
    std::shared_ptr<DomainBase> m_domain_;

   public:
    ContextBase(int argc = 0, char** argv = nullptr);

    template <typename TDomain>
    explicit ContextBase(std::shared_ptr<TDomain>&& domain)
        : m_domain_(std::dynamic_pointer_cast<DomainBase>(domain)) {}

    ~ContextBase() = default;

    DomainBase const& domain() const { return *m_domain_; }
};

template <typename TDomain>
class Context : public ContextBase {
   private:
    typedef Context<TDomain> ThisType;

   public:
    typedef TDomain DomainType;
    typedef Context<TDomain> ContextType;

    template <typename... Args>
    Context(Args&&... args) : ContextBase(std::make_shared<DomainType>(std::forward<Args>(args)...)) {}
    Context(int argc, char** argv) : ContextBase(argc, argv) {}
    ~Context() = default;

    DomainType const& domain() const { return static_cast<const DomainType&>(*m_domain_); }

    void initialize() {}
    void update(float dt) {}
    void update_bc(float dt) {}
};

}  // namespace simpla