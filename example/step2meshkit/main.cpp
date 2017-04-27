//
// Created by salmon on 16-11-27.
//

#include <simpla/utilities/Log.h>
#include <simpla/utilities/nTupleExt.h>

namespace simpla
{
void step2vtk(std::string const &input_name, std::string const &output_name = "");
}
using namespace simpla;

int main(int argc, char **argv)
{
    std::string input_file = "geqdsk.stp";
    std::string output_file = "geqdsk.vtk";
    if (argc <= 1)
    {
        std::cout << " Usage: " << argv[0] << " <input file> <output file> " << std::endl;
        exit(1);
    } else if (argc >= 1) { input_file = argv[1]; }
    else if (argc >= 2) { output_file = argv[2]; }

    std::cout << " input  :" << input_file << std::endl;
    std::cout << " output :" << output_file << std::endl;

    step2vtk(input_file, output_file);
}