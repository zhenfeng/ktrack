#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>
#include <iomanip>

#include "opencv_helpers.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <ctype.h>
#include <vector>
#include <iostream>

using namespace cv;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

namespace {

    Mat temp_rgb2gray;
    Mat temp_rgb_display;
    std::vector<Mat>  temp_rgb_channels;

    void matches2points(const vector<DMatch>& matches, const vector<KeyPoint>& kpts_train,
                        const vector<KeyPoint>& kpts_query,
                        vector<Point2f>& pts_train, vector<Point2f>& pts_query )
    {
      pts_train.clear();
      pts_query.clear();
      pts_train.reserve(matches.size());
      pts_query.reserve(matches.size());
      for (size_t i = 0; i < matches.size(); i++)
      {
        const DMatch& match = matches[i];
        pts_query.push_back(kpts_query[match.queryIdx].pt);
        pts_train.push_back(kpts_train[match.trainIdx].pt);
      }

    }

    double match(const vector<KeyPoint>& /*kpts_train*/, const vector<KeyPoint>& /*kpts_query*/,
                 DescriptorMatcher& matcher, const Mat& train, const Mat& query, vector<DMatch>& matches)
    {

      double t = (double)getTickCount();
      matcher.match(query, train, matches); //Using features2d
      return ((double)getTickCount() - t) / getTickFrequency();
    }


    Mat image;
    bool backprojMode = false;
    bool selectObject = false;
    int trackObject = 0;
    //bool showHist = true;
    Point origin;
    Rect selection;
    int vmin = 10, vmax = 256, smin = 30;

    void onMouse( int event, int x, int y, int, void* )
    {
        if( selectObject )
        {
            selection.x = MIN(x, origin.x);
            selection.y = MIN(y, origin.y);
            selection.width = std::abs(x - origin.x);
            selection.height = std::abs(y - origin.y);

            selection &= Rect(0, 0, image.cols, image.rows);
        }

        switch( event )
        {
        case CV_EVENT_LBUTTONDOWN:
            origin = Point(x,y);
            selection = Rect(x,y,0,0);
            selectObject = true;
            break;
        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if( selection.width > 0 && selection.height > 0 )
                trackObject = -1;
            break;
        }
    }

    Rect trackWindow;
    RotatedRect trackBox;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;


}


std::vector<Mat> chans;
cv::Mat          frame_8U;
void setup_blob_tracker()
{
    namedWindow( "Histogram", 1 );
    namedWindow( "CamShift Demo", 1 );
    setMouseCallback( "CamShift Demo", onMouse, 0 );
    createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
    createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
    createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );
    chans = std::vector<cv::Mat>(3);

}


int blobtrack_frame( const cv::Mat& frame, cv::Point2f& track_point ) {
// return code for tracking mode.
    track_point.x = frame.cols / 2;
    track_point.y = frame.rows / 2;

    if( frame.channels() == 3 ) {
        frame.copyTo(image);
    } else {
        frame.copyTo(frame_8U);
        chans[0] = frame_8U;
        cv::blur( frame_8U, chans[1], Size(3,3) );
        cv::blur( frame_8U, chans[2], Size(5,5) );
        chans[0] = frame_8U.clone();
        cv::merge( chans, image );
    }
    cvtColor(image, hsv, CV_BGR2HSV);

    if( trackObject ) // gets set after mouse input
    {
        int _vmin = vmin, _vmax = vmax;

        inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                Scalar(180, 256, MAX(_vmin, _vmax)), mask);
        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        if( trackObject < 0 )
        {
            Mat roi(hue, selection), maskroi(mask, selection);
            calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
            normalize(hist, hist, 0, 255, CV_MINMAX);

            trackWindow = selection;
            trackObject = 1;

            histimg = Scalar::all(0);
            int binW = histimg.cols / hsize;
            Mat buf(1, hsize, CV_8UC3);
            for( int i = 0; i < hsize; i++ )
                buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
            cvtColor(buf, buf, CV_HSV2BGR);

            for( int i = 0; i < hsize; i++ )
            {
                int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                rectangle( histimg, Point(i*binW,histimg.rows),
                           Point((i+1)*binW,histimg.rows - val),
                           Scalar(buf.at<Vec3b>(i)), -1, 8 );
            }
        }

        calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
        backproj &= mask;
        RotatedRect trackBox = CamShift(backproj, trackWindow,
                            TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));

        if( trackBox.size.height < 1 )
            trackBox.size.height = 2;
        else if( isnan( trackBox.size.height ) )
            trackBox.size.height = 100;
        if( trackBox.size.width < 1 )
            trackBox.size.width = 2;
        else if( isnan( trackBox.size.width ) )
            trackBox.size.width = 100;

        if( backprojMode )
            cvtColor( backproj, image, CV_GRAY2BGR );
        ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
        track_point.x = trackBox.center.x;
        track_point.y = trackBox.center.y;
    }

    if( selectObject && selection.width > 0 && selection.height > 0 )
    {
        Mat roi(image, selection);
        bitwise_not(roi, roi);
    }

    imshow( "CamShift Demo", image );
    imshow( "Histogram", histimg );

    return trackObject;
}


void frame2frame_feature_based(const cv::Mat &im1, const cv::Mat &im2,
                               Mat& warped, Mat& H, bool displayMatches) {

    int numMatches = 0;
    int minMatches = 20;
    int matchThresh = 100;
    vector<KeyPoint> kpts_1, kpts_2;
    vector<DMatch> matches_popcount;
    while( numMatches < minMatches ) {

        matchThresh = matchThresh / 2;

        FastFeatureDetector detector(matchThresh);
        BriefDescriptorExtractor extractor(32);

        detector.detect(im1, kpts_1);
        detector.detect(im2, kpts_2);

        Mat desc_1, desc_2;
        extractor.compute(im1, kpts_1, desc_1);
        extractor.compute(im2, kpts_2, desc_2);

        BruteForceMatcher<Hamming> matcher_popcount;
        match(kpts_1, kpts_2, matcher_popcount, desc_1, desc_2, matches_popcount);
        numMatches = std::min( kpts_1.size(), kpts_2.size() );
    }

    vector<Point2f> mpts_1, mpts_2;
    matches2points(matches_popcount, kpts_1, kpts_2, mpts_1, mpts_2);

    vector<uchar> outlier_mask;
    H = findHomography(mpts_2, mpts_1, RANSAC, 1, outlier_mask);

    if( displayMatches ) {
        Mat outimg;
        drawMatches(im2, kpts_2, im1, kpts_1, matches_popcount, outimg,
                    Scalar::all(-1), Scalar::all(-1),
                    reinterpret_cast<const vector<char>&> (outlier_mask));
        imshow("matches - popcount - outliers removed", outimg);
    }

    warpPerspective(im2, warped, H, im1.size());

}


void display_trackpoint( const cv::Mat& frame, const cv::Point2f& pt_xy ) {
    // draw a circle over the target and display it
    temp_rgb_display = Mat::zeros(frame.size(),CV_8UC3);
    cv::split( temp_rgb_display, temp_rgb_channels );
    cv::circle(temp_rgb_channels[1], pt_xy, 10, cv::Scalar(255,255,255),3,CV_AA);
    frame.copyTo(temp_rgb_channels[0]);
    frame.copyTo(temp_rgb_channels[2]);
    merge(temp_rgb_channels,temp_rgb_display);
    cv::imshow("track_frame", temp_rgb_display);
}

unsigned char*  make_quickset_frame( const Mat&  frame )
{
    //*ProcessFrameY8( short Width, short Height, unsigned char *VideoData )
    bool    isRGB   = (frame.channels() == 3);
    long int nBytes = frame.cols * frame.rows * (1+2*isRGB);
    unsigned char*  VideoData = (unsigned char*) malloc( nBytes );

    if( !isRGB )
        cv::cvtColor(frame,temp_rgb2gray,CV_RGB2GRAY);
    else
        temp_rgb2gray = frame;

    memcpy( VideoData, temp_rgb2gray.ptr<unsigned char>(0), nBytes );

    return VideoData;
}

void get_quickset_frame( unsigned char*  VideoData, Mat& frame)
{
    int     nchans  = frame.channels();
    std::cout << "nchannels: " << nchans;
    bool    isRGB   = (nchans == 3);
    long int nBytes = frame.cols * frame.rows * (1+2*isRGB);

    assert( nBytes > 16 ); // all good ? check the allocation of frame otherwise

    memcpy( frame.ptr<unsigned char>(0), VideoData, nBytes );

}


