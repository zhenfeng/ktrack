/*
 * file_io.cpp
 *
 *  Created on: Nov 27, 2010
 *      Author: ethan
 *  modified by peter, Spring 2011
 */

#ifndef FILE_IO_CPP_ROBOTVIEW_JNI
#define FILE_IO_CPP_ROBOTVIEW_JNI



#include <string>
#include <vector>

void getFilePrefixes(const char * dir, const std::string& extension,
                     std:: vector<std::string>& files);
int getFileNum(const char * dir, const std::string& extension);

void lsFilesOfType(const char * dir, const std::string& extension,
                   std::vector<std::string>& files);



#endif /* FILE_IO_CPP_ */
