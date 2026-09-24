#include<iostream>

#include"utils.h"
#include"flow.h"

/***************************************************************************************************
 *                                      编程荣辱观 · Coding Honor Code                              *
 ***************************************************************************************************
 *                                                                                                 *
 * 以动手实践为荣，以只看不练为耻。       Honor in hands-on practice; shame in passive watching.      *
 * 以打印日志为荣，以出错不报为耻。            Honor in logging; shame in silent failures.            *
 * 以局部变量为荣，以全局变量为耻。           Honor in local scope; shame in global sprawl.           *
 * 以自动测试为荣，以手工测试为耻。        Honor in automated testing; shame in manual poking.        *
 * 以代码重用为荣，以复制粘贴为耻。               Honor in reuse; shame in copy-paste.                *
 * 以多态应用为荣，以分支判断为耻。        Honor in polymorphism; shame in excessive branching.       *
 * 以定义常量为荣，以魔法数字为耻。         Honor in named constants; shame in magic numbers.         *
 * 以总结思考为荣，以不求甚解为耻。   Honor in reflection and understanding; shame in shallow coding. *
 *                                                                                                 *
 ***************************************************************************************************/


int main(int argc, char* argv[]) {
    
    MainFlow mainflow(argv[1], argv[2]);
    mainflow.runflow();

    return 0;
}



