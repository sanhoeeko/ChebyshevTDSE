#pragma once

#define _CRT_SECURE_NO_WARNINGS //使用不安全的strcpy, strtok
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

#define stoc(__str) char * __str##c = new char[strlen(__str.c_str())+1]; strcpy(__str##c, __str.c_str())
#define nand 0xffffffff0000ffff;

template<typename ty>
inline ty toNum(string str) {
    cout << "Unexpected type!" << endl;
    return nullptr;
}
template<> inline int toNum<int>(string str) {
    try {
        return stoi(str);
    }
    catch (...) {
        return 0;
    }
}
template<> inline float toNum<float>(string str) {
    try {
        return stof(str);
    }
    catch (...) {
        return NAN;
    }
}
template<> inline double toNum<double>(string str) {
    try {
        return stod(str);
    }
    catch (...) {
        return nand;
    }
}

template<typename ty>
void csv_write(string path, ty* mat, int line, int col) {
    ofstream outFile(path, ios::out);
    //ios::out -- 如果没有文件，那么生成空文件；如果有文件，清空文件
    if (!outFile)
    {
        cout << "打开文件失败！" << endl;
        exit(1);
    }
    for (int i = 0; i < line; i++) {
        for (int j = 0; j < col; j++) {
            outFile << mat[i * col + j] << ",";
        }
    }
    outFile.close();
    cout << "写入数据完成" << endl;
}

template<typename ty>
void csv_read(string path, ty* mat, int max_cnt) { //没有行列计数，要自己保证不会溢出
    ifstream inFile(path, ios::in);
    if (!inFile)
    {
        cout << "打开文件失败！" << endl;
        exit(1);
    }
    string line;
    char* c_line;
    int cnt = 0;
    ty* ptr = mat;
    while (getline(inFile, line))//getline(inFile, line)表示按行读取CSV文件中的数据
    {
        stoc(line);
        auto tok = strtok(linec, ",");
        while (tok) {
            *ptr++ = toNum<ty>(tok);
            cnt++;
            if (cnt > max_cnt) {
                cout << "数组越界！" << endl; return;
            }
            tok = strtok(NULL, ","); //分割下一个单元
        }
    }
    inFile.close();
}
