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
class Pool { //真正的内存都在这里
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
			ptrs[i].idx = i; //为了防止泄漏，必须立即分配内存
#if CUDA_AVAIL
			gram(ptrs[i].loc, ty, (int64_t)W * H * 2); //可能改变loc的地址
#else
			ptrs[i].loc = cram(ty, (int64_t)W * H * 2);
#endif
			occupied[i] = false;
		}
	}
	Pool(const Pool& x) {
		printf("Pool对象不可复制！");
	}
	~Pool() {
		//free(head); free(occupied); free(ptrs);
	}
	iptr<ty> apply() { //申请指针，具体写入方法由操作方控制
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
	void del(iptr<ty>& ptr) { //仅仅标记为未占用
		occupied[ptr.idx] = false;
		__show__
	}
	void showOccupation() {
		for (int i = 0; i < max_num; i++) {
			printf(occupied[i] ? "■" : "□");
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