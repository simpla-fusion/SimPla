
#include "frc.hpp"

#include "simpla/utils/logger.hpp"

void FRC::initialize() {
    LOGGER << __FUNCTION__ << std::endl;

    B.set_value(1.0);
    J.set_value(2.0);
}
void FRC::update(float dt) {}
void FRC::update_bc(float dt) {}