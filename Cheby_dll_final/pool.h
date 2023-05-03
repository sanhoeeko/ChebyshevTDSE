#pragma once

#include<vector>
#include<stdlib.h>
#include<stdio.h>
#include"param.h"

#define DEBUG false

#if DEBUG
#define __show__ showOccupation();
#else
#define __show__
#endif

#define cram(__ty,__n) (__ty*)calloc((__n),sizeof(__ty))
#define gram(__ptr,__ty,__n) cudaMalloc((void**)&__ptr,(__n)*sizeof(__ty))

template<typename ty>
struct iptr {
	ty* loc;
	int idx;
};

template<typename ty>
iptr<ty> inull() {
	iptr<ty> res = { NULL, -1 };
	return res;
}
#define IPTR_NULL inull<ty>()

template<typename ty>
class Pool { //�������ڴ涼������
public:
	ty* head;
	iptr<ty>* ptrs;
	bool* occupied;
	int max_num;
	int type_size;

	Pool(int max_num) {
		this->max_num = max_num;
		head = cram(ty, max_num);
		occupied = cram(bool, max_num);
		ptrs = cram(iptr<ty>, max_num);
		for (int i = 0; i < max_num; i++) { 
			ptrs[i].idx = i; //Ϊ�˷�ֹй©���������������ڴ�
#if CUDA_AVAIL
			gram(ptrs[i].loc, ty, (int64_t)W * H * 2); //���ܸı�loc�ĵ�ַ
#else
			ptrs[i].loc = cram(ty, (int64_t)W * H * 2);
#endif
			occupied[i] = false;
		}
	}
	Pool(const Pool& x) {
		printf("Pool���󲻿ɸ��ƣ�");
	}
	~Pool() {
		//free(head); free(occupied); free(ptrs);
	}
	iptr<ty> apply() { //����ָ�룬����д�뷽���ɲ���������
		for (int i = 0; i < max_num; i++) {
			if (!occupied[i]) {
				occupied[i] = true;
				__show__
				return ptrs[i];
			}
		}
		printf("Memory pool error!\n");
		return IPTR_NULL;
	}
	void del(iptr<ty>& ptr) { //�������Ϊδռ��
		occupied[ptr.idx] = false;
		__show__
	}
	void showOccupation() {
		for (int i = 0; i < max_num; i++) {
			printf(occupied[i] ? "��" : "��");
		}
		printf("\n");
	}
	void clear_all() {
		for (int i = 0; i < max_num; i++) {
			occupied[i] = false;
		}
	}
	void free_all() {
		for (int i = 0; i < max_num; i++) {
#if CUDA_AVAIL
			cudaFree(ptrs[i].loc);
#else 
			safe_free(ptrs[i].loc);
#endif
		}
	}
};