#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using std::cout;
using std::endl;

struct Options {
    int vid;
};

unsigned char*  make_quickset_frame( const cv::Mat&  frame );

void get_quickset_frame( unsigned char*  VideoData, cv::Mat& frame);

void display_trackpoint( const cv::Mat& frame, const cv::Point2f& pt_xy );

void frame2frame_feature_based(const cv::Mat &im1, const cv::Mat &im2,
                               cv::Mat& warped, cv::Mat& H, bool displayMatches);

int blobtrack_frame( const cv::Mat& frame, cv::Point2f& track_point);

void setup_blob_tracker();

struct pftracker {

    cv::Mat likelihood;
    cv::Mat Lshow;

    cv::Mat background_image;
    cv::Mat current_image;
    cv::Mat difference_image;
    cv::Mat previous_image;

    cv::Mat curr_8U;
    cv::Mat prev_8U;
    cv::Mat bgnd_8U;

    cv::Mat frame_rgb;
    std::vector<cv::Mat>  chans;

    void process_frame( const cv::Mat& frame, cv::Point2f&  track_point ){
        if( background_image.empty() ) {
            cout << "attempting to convert frame, nchans = " << frame.channels() << std::endl;
            frame.convertTo(background_image,CV_32FC3);
            track_point.x = frame.cols / 2;
            track_point.y = frame.rows / 2;
            frame.convertTo(previous_image,CV_32FC3);
            frame.copyTo(prev_8U);
            likelihood = cv::Mat::zeros(frame.size(),CV_32FC3);
            setup_blob_tracker();
            chans = std::vector<cv::Mat>(3);
            return;
        }
        frame.convertTo(current_image,CV_32FC3); // copy to be safe
        frame.copyTo(curr_8U);
        assert( curr_8U.type() == CV_8UC3 );
        int track_mode = blobtrack_frame(curr_8U,track_point);
        cout << "track mode: " << track_mode << endl;

    }
    void process_frame_2( const cv::Mat& frame, cv::Point2f&  track_point ){

        // combine:
        //     1) difference image
        //     2) time-averaged background
        //     3) brightness
        //             into a useful likelihood function

        if( background_image.empty() ) {
            frame.convertTo(background_image,CV_32F);
            track_point.x = frame.cols / 2;
            track_point.y = frame.rows / 2;
            frame.convertTo(previous_image,CV_32F);
            frame.copyTo(prev_8U);
            likelihood = cv::Mat::zeros(frame.size(),CV_32F);
            setup_blob_tracker();
            chans = std::vector<cv::Mat>(3);
            return;
        }
        frame.convertTo(current_image,CV_32F); // copy to be safe
        frame.copyTo(curr_8U);

        background_image = background_image * 0.9 + current_image * 0.1;
        background_image.convertTo(bgnd_8U,CV_8U);
        // Click & Track
        chans[0] = curr_8U;
        chans[1] = prev_8U;
        chans[2] = bgnd_8U;
        cv::merge(chans,frame_rgb);
        int track_mode = blobtrack_frame(curr_8U,track_point);

        if( track_mode == 0 )
        {  // no clicks yet

            ////////////////////////////////////// Detect/Track Mixture
            cv::Mat warped, H;
            frame2frame_feature_based( curr_8U,prev_8U, warped, H, false /*display matches*/ );
            warped.convertTo(previous_image,CV_32F);

            bool detection_mode = false; // TODO: set this externally
            if( detection_mode ) { // "Detection": we're not moving, but looking for something in steady background
                background_image = background_image * 0.95 + current_image * 0.05;
                cv::absdiff( current_image*(1.0/255.0),previous_image*(1.0/255.0),difference_image );
            } else {               // moving! compared affine warped to previous
                background_image = background_image * 0.7 + current_image * 0.3;
                cv::absdiff( current_image*(1.0/255.0),previous_image*(1.0/255.0),difference_image);
            }

            likelihood = likelihood * 0.7 + difference_image * 0.3;
            cv::pow( difference_image,4.0,likelihood );
            double dmin, dmax;
            cv::minMaxLoc( likelihood, &dmin, &dmax );
            cout << "min,max likelihood: " << dmin << ", " << dmax << endl;
            likelihood = (likelihood - dmin) * ( 1.0 / ( dmax - dmin ) );

            // 'pseudo single particle': compute weighted mean over the coords
            float px = 0.0;
            float py = 0.0;
            //float pixvar = 200.0;
            //likelihood.at<float>(i,j) *= exp( -( pow(track_point.x-j,2.0)+pow(track_point.y-i,2.0) )/pixvar );
            for( int i = 0; i < frame.rows; i++ ) {
                for( int j = 0; j < frame.cols; j++ ) {
                   // warped affine frame 2 frame is zero or we're at the edge
                   if( i < 8 || j < 8 || i > (frame.rows-8) || j > (frame.cols-8) ||
                                              previous_image.at<float>(i,j) < 5.0 ) {
                     likelihood.at<float>(i,j) = 0;
                   }
                   px = px + j * likelihood.at<float>(i,j);
                   py = py + i * likelihood.at<float>(i,j);
                }
            }
            float sumL    = cv::sum( likelihood )[0]; // normalize and store x,y points
            track_point.x = px / (sumL+1e-2);
            track_point.y = py / (sumL+1e-2);
        }

        current_image.copyTo(previous_image); // bag the previous image, float & int
        curr_8U.copyTo(prev_8U);

    }

};
