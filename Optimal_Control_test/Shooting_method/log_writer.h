#pragma once
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using namespace std;
#ifndef LOGER_
#define LOGER_
class LogHandler {
public:
    LogHandler(string filename, string fileHeader) {
        fileName_ = filename;
        fileHeader_ = fileHeader;
        OpenFile();
    }
    ~LogHandler() {
        CloseFile();
        cout << "filehandle released:" << fileName_ << endl;
    }

    void OpenFile() {
        fileObj_.open(fileName_);
        fileObj_ << fileHeader_ << endl;
    }
    void CloseFile() { fileObj_.close(); }

    void WriteString(string str) { fileObj_ << str << endl; }
    void WriteData(vector<double> &data) {
        for (auto p : data) {
            fileObj_ << p << " ";
        }
        fileObj_ << endl;
    }
    void WriteData(list<vector<double>> &data) {
        for (auto rowdata : data) {  // output one row data
            for (auto p : rowdata) {
                fileObj_ << p << " ";
            }
            fileObj_ << endl;
        }
    }
    // element
    string fileName_;
    string fileHeader_;
    ofstream fileObj_;
};

#endif