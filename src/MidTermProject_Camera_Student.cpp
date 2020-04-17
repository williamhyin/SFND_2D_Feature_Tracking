/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    vector<double> det_record_time(10);
    vector<int> det_record_num(10);
    vector<double> des_record_time(10);
    vector<double> mat_record_time(9);
    vector<int> mat_record_num(9);

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);


        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        // remove old frame in left side of dataBuffer and add new frame in right side
        if (dataBuffer.size() > dataBufferSize) {
            dataBuffer.erase(dataBuffer.begin());
        }
        dataBuffer.push_back(frame);

//        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

//        string detectorType = "SHITOMASI";
//         string detectorType = "HARRIS";
         string detectorType = "FAST";
//         string detectorType = "BRISK";
//         string detectorType = "ORB";
//         string detectorType = "AKAZE";
//         string detectorType = "SIFT";

        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        // Time count for detector
        double t_det = (double) cv::getTickCount();
        if (detectorType.compare("SHITOMASI") == 0) {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        } else if (detectorType.compare("HARRIS") == 0) {
            detKeypointsHarris(keypoints, imgGray, false);

        } else if (detectorType.compare("BRISK") == 0 || detectorType.compare("SIFT") == 0 ||
                   detectorType.compare("AKAZE") == 0 || detectorType.compare("ORB") == 0 ||
                   detectorType.compare("FAST") == 0) {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        } else {
            throw invalid_argument(detectorType + " is not supported");
        }

        t_det = ((double) cv::getTickCount() - t_det) / cv::getTickFrequency();
        t_det = 1000 * t_det / 1.0;
        cout << detectorType << " detection in " << t_det << " ms" << endl;


        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        vector<cv::KeyPoint> insidePoints;
        if (bFocusOnVehicle) {
            for (auto keypt:keypoints) {
                bool isinside = vehicleRect.contains(keypt.pt);
                if (isinside) {
                    insidePoints.push_back(keypt);
                }
            }
            keypoints = insidePoints;
        }
        cout << "Number of keypoints on the preceding vehicle:" << keypoints.size() << endl;
        det_record_num.push_back(keypoints.size());
        det_record_time.push_back(t_det);

        // draw figure of the keypoints on the preceding vehicle
        if (false) {
            cv::Mat visImage = img.clone();
            cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            string windowName = "Keypoints on the preceding vehicle";
            cv::namedWindow(windowName);
            imshow(windowName, visImage);
            cv::waitKey(0);
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;
            if (detectorType.compare("SHITOMASI") ==0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
//        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRDETECTORIPTORS */


        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;

        // BRIEF, ORB, FREAK, AKAZE, SIFT
        string descriptorType = "BRIEF";
//         string descriptorType = "ORB";
//         string descriptorType = "FREAK";
//         string descriptorType = "AKAZE";
//         string descriptorType = "SIFT";

        double t_des = (double) cv::getTickCount();
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors,
                      descriptorType);
        t_des = ((double) cv::getTickCount() - t_des) / cv::getTickFrequency();
        t_des = 1000 * t_des / 1.0;
        cout << descriptorType << " descriptor extraction in " << t_des << " ms" << endl;

        des_record_time.push_back(t_des);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

//        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;

            string matcherType = "MAT_BF";        // brute force (BF)
//             string matcherType = "MAT_FLANN";     // Fast Library for Approximate Nearest Neighbors (FLANN)

            // BINARY descriptors :BRISK, BRIEF, ORB, FREAK, and AKAZE
            // HOG descriptors : SIFT (and SURF and GLOH, all patented).
            string descriptorclass{};
            if (descriptorType.compare("SIFT")==0) {
                descriptorclass = "DES_HOG";

            } else {
                descriptorclass = "DES_BINARY";
            }

//             string selectorType = "SEL_NN"; // choose nearest neighbors (NN)
            string selectorType = "SEL_KNN";   // k nearest neighbors (KNN)


            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            double t_mat = (double) cv::getTickCount();
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorclass, matcherType, selectorType);
            t_mat = ((double) cv::getTickCount() - t_mat) / cv::getTickFrequency();
            t_mat = 1000 * t_mat / 1.0;

            cout << matcherType << " descriptor matching in " << t_mat << " ms" << endl;
            cout<<endl;
            mat_record_num.push_back(matches.size());
            mat_record_time.push_back(t_mat);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

//            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = true;
            if (bVis) {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cout << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images
    double ave_det_time=double(accumulate(det_record_time.begin(), det_record_time.end(), 0.) / 10.0);
    double ave_det_num=double(accumulate(det_record_num.begin(), det_record_num.end(), 0.) / 10.0);
    double ave_des_time=double(accumulate(des_record_time.begin(), des_record_time.end(), 0.) / 10.0);
    double ave_mat_time=double(accumulate(mat_record_time.begin(), mat_record_time.end(), 0.) / 9.0);
    double ave_mat_num=double(accumulate(mat_record_num.begin(), mat_record_num.end(), 0.) / 9.0);
    double ave_total_time=ave_des_time+ave_det_time+ave_mat_time;

    cout << "Average detection time: " <<ave_det_time<< endl;
    cout << "Average detection num: " <<ave_det_num<< endl;
    cout << "Average description time: "<< ave_des_time << endl;
    cout << "Average matching time: " << ave_mat_time<< endl;
    cout << "Average matching num: " << ave_mat_num<< endl;
    cout << "Average total time: " << ave_total_time<< endl;

    return 0;
}
