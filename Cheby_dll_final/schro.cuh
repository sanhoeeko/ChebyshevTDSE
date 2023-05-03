#pragma once

#define def extern "C"

#include"param.h"

def inline void gpu_five_point(real* src, real* dst, real a);
def inline void host_five_point(real* src, real* dst, real a); //cpu
def inline void naive_five_point(real* src, real* dst, real a);
def inline void host_naive_five_point(real* src, real* dst, real a); //cpu
def inline void gpu_add(real* dst_a, real* b);
def inline void gpu_minus(real* dst_a, real* b);
def inline void gpu_add_i(real* dst_a, real* b);
def inline void gpu_minus_i(real* dst_a, real* b);
def inline void gpu_mulnum(real* dst_a, real coef);
def inline void host_add(real* dst_a, real* b); //cpu
def inline void host_minus(real* dst_a, real* b); //cpu
def inline void host_add_i(real* dst_a, real* b); //cpu
def inline void host_minus_i(real* dst_a, real* b); //cpu
def inline void host_mulnum(real* dst_a, real coef); //cpu
def inline void gpu_free(real* obj);
def inline void gpu_copy(real* src, real* dst);
def inline void gpu_output(real* src, real* dst);
def inline void gpu_upload(real* src, real* dst);
def inline void gpu_wall(real* dst, int* wall);
def inline void host_wall(real* dst, int* wall);