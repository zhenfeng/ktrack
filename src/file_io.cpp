/*
 * file_io.cpp
 *
 *  Created on: Nov 27, 2010
 *      Author: ethan
 *  modified by peter, Spring 2011
 */

#include "file_io.h"
#include <sys/types.h>
#include <dirent.h>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

/**  given a "dir" as string and ending extension, put name of files
     into the vector string. vector is sorted lexicographically.
  */
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
    std::sort(files.begin(), files.end());
}

/**  given an ending extension, put prefix name of files
     into the vector string. vector is sorted lexicographically.
     e.g. dir = "/1941" contains smolensk.jpg, sevastopol.jpg, guderian.jpg
     and extension = ".jpg" gives files as [guderian, sevastopol, smolensk] */
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

/** this function is unknown and possibly incompatible with the changes
    to those getting file lists.
  */
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
