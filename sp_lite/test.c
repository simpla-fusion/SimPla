//
// Created by salmon on 16-7-16.
//

#include <stdio.h>
#include <stdlib.h>
struct Foo
{
    int data[5];
};

//struct U
//{
//
//    int data[10]__attribute__ ((aligned(16)));
//};

int main(int argc, char **argv)
{
    struct Foo f;
//    struct U u[5];
    int a[5] = {7, 6, 4, 5, 4};


    memcpy(&f, a, sizeof(int) * 5);
    printf("sizeof(Foo) \t= %d ,\t addr= 0x%x\n", sizeof(struct Foo), f.data);
    for (int i = 0; i < 5; ++i)
    {
        printf("[%d] = %d  \n", i, f.data[i]);
    }
//    printf("sizeof(U) \t= %d ,\t 0x%x addr= 0x%x\n", sizeof(struct U), &u[0], u[0].data);
//    printf("sizeof(U) \t= %d ,\t 0x%x addr= 0x%x\n", sizeof(struct U), &u[1], u[1].data);
//    printf("sizeof(U) \t= %d ,\t 0x%x addr= 0x%x\n", sizeof(struct U), &u[2], u[2].data);

}
