//
// Created by salmon on 16-12-26.
//

#include "../Algebra.h"
#include "../Field.h"
#include "../DummyMesh.h"

using namespace simpla;

int main(int argc, char **argv)
{
    DummyMesh m;

    Field<Real, DummyMesh> f(&m);

    f = 0;
}