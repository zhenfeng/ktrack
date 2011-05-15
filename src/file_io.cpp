/*
 * file_io.cpp
 *
 *  Created on: Nov 27, 2010
 *      Author: ethan
 */

#include "file_io.h"
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

void lsFilesOfType(const char * dir, const string& extension,
        vector<string>& files) {
    files.clear();
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir)) == NULL) {
        return;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name(dirp->d_name);
        size_t pos = name.find(extension);
        if (pos != std::string::npos) {
            files.push_back(name);
        }
    }
    closedir(dp);
}

void getFilePrefixes(const char * dir, const string& extension,
        vector<string>& files) {

    files.clear();
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir)) == NULL) {
        return;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name(dirp->d_name);
        size_t pos = name.find(extension);
        if (pos != std::string::npos) {
            files.push_back(name.substr(0, pos));
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
}
int getFileNum(const char * dir, const std::string& extension) {
    int count = 0;
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir)) == NULL) {
        return 0;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name(dirp->d_name);
        size_t pos = name.find(extension);
        if (pos != std::string::npos) {
            count++;
        }
    }
    closedir(dp);
    return count;
}
