#include "contour.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::vector;
using std::cout;
using std::endl;
using namespace vrcl;
using namespace cv;

namespace {
  const static string PrintVerbose = "VerboseKSegUtils";
}

namespace  ktrack
{

  void waterMark(const std::string& text, Mat & img)
  {
    int baseline = 0;
    Size textSize = getTextSize(text, CV_FONT_HERSHEY_PLAIN, 3, 2, &baseline);
    Point textOrg((img.cols - textSize.width) / 2, (img.rows + textSize.height) / 2);
    putText(img, text, textOrg, CV_FONT_HERSHEY_PLAIN, 3, Scalar::all(1), 2, 8);

  }



} // end namespace


