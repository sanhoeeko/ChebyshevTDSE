#pragma once

#include<stdlib.h>
#include<math.h>
#define EPS 1e-10
#define CEIL 1e4

class BesselI_array { //已测试
public:
	double* val;
	BesselI_array(double x, int m) {
		val = (double*)malloc(sizeof(double) * m);
		double* ptr = val + m - 1;
		*ptr-- = 0;
		*ptr-- = EPS;
		int n = m - 2;
		while (ptr >= val) {
			*ptr = *(ptr + 1) * 2 * n / x + *(ptr + 2);
			if (*ptr > CEIL) {
				double* renormalizer = val + m - 1;
				while (renormalizer >= ptr) {
					*renormalizer /= CEIL;
					renormalizer--;
				}
			}
			n--; ptr--;
		}
		double s = val[0] / 2;
		bool flag = false;
		for (int i = 2; i < m; i+=2) { //I的归一化：I0-2I2+2I4-2I6+...=1
			s += flag? val[i] : -val[i];
			flag = !flag;
		}
		double coef = 0.5 / s; //乘法比除法快
		for (int i = 0; i < m; i++) {
			val[i] *= coef;
		}
	}
	~BesselI_array() {
		free(val);
	}
};

class BesselJ_array { //已测试
public:
	double* val;
	BesselJ_array(double x, int m) {
		val = (double*)malloc(sizeof(double) * m);
		double* ptr = val + m - 1;
		*ptr-- = 0;
		*ptr-- = EPS;
		int n = m - 2;
		while (ptr >= val) {
			*ptr = *(ptr + 1) * 2 * n / x - *(ptr + 2);
			if (*ptr > CEIL) {
				double* renormalizer = val + m - 1;
				while (renormalizer >= ptr) {
					*renormalizer /= CEIL;
					renormalizer--;
				}
			}
			n--; ptr--;
		}
		double s = val[0] * val[0] / 2;
		for (int i = 1; i < m; i++) { //I的归一化：I0-2I2+2I4-2I6+...=1
			s += val[i] * val[i];
		}
		double coef = sqrt(0.5 / s); //乘法比除法快
		for (int i = 0; i < m; i++) {
			val[i] *= coef;
		}
	}
	~BesselJ_array() {
		free(val);
	}
};