#include <simpla/core/context.hpp>
#include <simpla/core/domain.hpp>
#include <simpla/core/field.hpp>
#include <simpla/utils/logger.hpp>
using namespace simpla;

class FRC : public Context<Domain<1>> {
   public:
    template <typename... Args>
    FRC(Args... args) : ContextType(std::forward<Args>(args)...) {
        VERBOSE << __FUNCTION__ << std::endl;
    }

    ~FRC() = default;

    Field<double, 3, DomainType> B{this};
    Field<double, 3, DomainType> J{this};
    Field<double, 0, DomainType> n{this};
    Field<double, 0, DomainType> T{this};

    void initialize();
    void update(float dt);
    void update_bc(float dt);
};