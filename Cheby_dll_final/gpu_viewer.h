#pragma once

#include"schro.cuh"
#include<stdint.h>
#include<stdlib.h>
#include<iostream>
#include "mat2syms.h"
using namespace std;
#define cram(__ty,__n) (__ty*)calloc((__n),sizeof(__ty))
#define eidx(__i,__j) ((__i)*H+(__j))*2
#define oidx(__i,__j) eidx(__i,__j)+1

void c_show(real* mat) {
	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) {
			cout << mat[eidx(i, j)] << "," << mat[oidx(i, j)] << "  ";
		}
		cout << endl;
	}
	cout << endl;
}

void g_show(real* gpu) {
	real* temp = cram(real, (int64_t)W * H * 2);
	gpu_output(gpu, temp);
	c_show(temp);
	free(temp);
}

void rho_show(real* mat, real max, real min, int* wall) {
	real rho;
	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) {
			if (wall[i * H + j]) {
				cout << "¡ö"; continue;
			}
			rho = mat[eidx(i, j)] * mat[eidx(i, j)] + mat[oidx(i, j)] * mat[oidx(i, j)];
			cout << toChar(rho, max, min) << " ";
		}
		cout << endl;
	}
	cout << endl;
}