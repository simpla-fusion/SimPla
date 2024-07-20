#include "context.hpp"

#include "simpla/utils/logger.hpp"

namespace simpla {
ContextBase::ContextBase(int argc, char** argv) : m_domain_(nullptr) { VERBOSE; }
}  // namespace simpla