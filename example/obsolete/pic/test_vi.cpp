/**
 * @file test_vi.cpp
 * @author salmon
 * @date 2016-01-27.
 */

#include <vector>
#include <iostream>
#include <stdlib.h>

#include "../../core/io/IO.h"

using namespace simpla;

int main(int argc, char **argv)
{

    std::vector<double> x1, v1;

    std::vector<double> x2, v2;

    double x0 = 0;

    double v0 = 0.1;

    size_t num_of_step = (argc > 1) ? static_cast<size_t>(atoi(argv[1])) : 1000000L;

    double Delta = (argc > 2) ? atof(argv[2]) : 0.1;

    double Delta2 = Delta * Delta;
    double Delta3 = Delta * Delta2;
    double Delta4 = Delta2 * Delta2;
    double Delta5 = Delta * Delta4;


    x1.resize(num_of_step + 1);
    v1.resize(num_of_step + 1);
    x2.resize(num_of_step + 1);
    v2.resize(num_of_step + 1);


    x1[0] = x0;
    v1[0] = v0;

    x2[0] = x0;
    v2[0] = v0;


    for (int n = 0; n < num_of_step; ++n)
    {

        v1[n + 1] = 0.5 * (7 * Delta4 * v1[n] + 45 * Delta3 * x1[n] - 192 * Delta2 * v1[n] - 420 * Delta * x1[n] +
                           420 * v1[n]) / (Delta4 + 9 * Delta2 + 210);

        x1[n + 1] = 0.5 * (Delta5 * v1[n] + 7 * Delta4 * x1[n] - 52 * Delta3 * v1[n] - 192 * Delta2 * x1[n] +
                           420 * Delta * v1[n] + 420 * x1[n]) / (Delta4 + 9 * Delta2 + 210);
//#ifdef DEBUG
//        std::cout << "n = " << n + 1 << " x = " << x[n + 1] << " k = " << kx[n + 1] << std::endl;
//#endif


    }


    simpla::io::cd("test_vi.h5:/");

    std::cout << simpla::io::write("x1", x1) << std::endl;

    std::cout << simpla::io::write("v1", v1) << std::endl;

    std::cout << simpla::io::write("x2", x2) << std::endl;

    std::cout << simpla::io::write("v2", v2) << std::endl;

    std::cout << "Done!" << std::endl;
}