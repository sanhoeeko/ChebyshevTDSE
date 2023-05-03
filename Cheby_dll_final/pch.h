// pch.h: 这是预编译标头文件。
// 下方列出的文件仅编译一次，提高了将来生成的生成性能。
// 这还将影响 IntelliSense 性能，包括代码完成和许多代码浏览功能。
// 但是，如果此处列出的文件中的任何一个在生成之间有更新，它们全部都将被重新编译。
// 请勿在此处添加要频繁更新的文件，这将使得性能优势无效。

#ifndef PCH_H
#define PCH_H

#define def extern "C" _declspec(dllexport)
#define _CRT_SECURE_NO_WARNINGS

// 添加要在此处预编译的标头
#include "framework.h"

def double* main_proj_init();
def double main_proj_single();
def void end_test();
def void test();
def void set_timeSpan(double t);
def void set_iter(int n);
def void set_momentum(double p);
def void set_sigma(double s);
def void seeParams();

#endif //PCH_H
