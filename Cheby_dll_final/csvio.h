#pragma once

#define _CRT_SECURE_NO_WARNINGS //ʹ�ò���ȫ��strcpy, strtok
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
    //ios::out -- ���û���ļ�����ô���ɿ��ļ���������ļ�������ļ�
    if (!outFile)
    {
        cout << "���ļ�ʧ�ܣ�" << endl;
        exit(1);
    }
    for (int i = 0; i < line; i++) {
        for (int j = 0; j < col; j++) {
            outFile << mat[i * col + j] << ",";
        }
    }
    outFile.close();
    cout << "д���������" << endl;
}

template<typename ty>
void csv_read(string path, ty* mat, int max_cnt) { //û�����м�����Ҫ�Լ���֤�������
    ifstream inFile(path, ios::in);
    if (!inFile)
    {
        cout << "���ļ�ʧ�ܣ�" << endl;
        exit(1);
    }
    string line;
    char* c_line;
    int cnt = 0;
    ty* ptr = mat;
    while (getline(inFile, line))//getline(inFile, line)��ʾ���ж�ȡCSV�ļ��е�����
    {
        stoc(line);
        auto tok = strtok(linec, ",");
        while (tok) {
            *ptr++ = toNum<ty>(tok);
            cnt++;
            if (cnt > max_cnt) {
                cout << "����Խ�磡" << endl; return;
            }
            tok = strtok(NULL, ","); //�ָ���һ����Ԫ
        }
    }
    inFile.close();
}
