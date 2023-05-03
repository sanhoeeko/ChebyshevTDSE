#pragma once

#include<stdio.h>
#include<stdlib.h>

const char __gray_str[] =
"@B%8&WM#ZO0QLCJUYXlI1*oahkbdpqwmzcvunxrjfti!+/\\|(){}[]?-_~<>;:,\"^`'. "; //64位

inline char toChar(double val, double max, double min) {
	if (val <= min) {
		return __gray_str[63];
	}
	else if (val >= max) {
		return __gray_str[0]; 
	}
	else {
		double x = (val - min) / (max - min);
		int idx = 63 - x * 64;
		return __gray_str[idx];
	}
}

void toSymPic(double* mat, char* dst,int m,int n,double max,double min) {
	double _span = 1 / (max - min);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			int loc = i * n + j;
			if (mat[loc] <= min) {
				dst[loc] = __gray_str[63]; continue;
			}
			else if (mat[loc] >= max) {
				dst[loc] = __gray_str[0]; continue;
			}
			else {
				double x = (mat[loc] - min) * _span;
				int idx = 63 - x * 64;
				dst[loc] = __gray_str[idx];
			}
		}
	}
}

void showSymPic(char* syms, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			putchar(syms[i * H + j]);
		}
		putchar('\n');
	}
	putchar('\n');
}