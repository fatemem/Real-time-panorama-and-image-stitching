#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <iostream>
#include "opencv2/core/cuda.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>



using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;

bool first = true;
Mat homo;
Mat paranoma(Mat image1, Mat image2) {
	Size size(1024, 780);

	cv::resize(image1, image1, size);
	cv::resize(image2, image2, size);

	Mat gray_image1;
	Mat gray_image2;

	//Covert to Grayscale
	cvtColor(image1, gray_image1, cv::COLOR_BGRA2GRAY);
	cvtColor(image2, gray_image2, cv::COLOR_BGRA2GRAY);




	


	//Find the Homography Matrix
	if (first) {
		//--Step 1 : Detect the keypoints using SURF Detector
		auto start = std::chrono::high_resolution_clock::now();
		SURF_CUDA surf= cv::cuda::SURF_CUDA::SURF_CUDA(400,4,3,false, 0.01f,false);
		GpuMat img1, img2;
		img1.upload(gray_image1);
		img2.upload(gray_image2);
		// detecting keypoints & computing descriptors
		GpuMat keypoints1GPU, keypoints2GPU;
		GpuMat descriptors1GPU, descriptors2GPU;
		surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
		surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

		// matching descriptors
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
		vector<DMatch> matches;
		matcher->match(descriptors1GPU, descriptors2GPU, matches);

		// downloading results
		vector<KeyPoint> keypoints1, keypoints2;
		vector<float> descriptors1, descriptors2;
		surf.downloadKeypoints(keypoints1GPU, keypoints1);
		surf.downloadKeypoints(keypoints2GPU, keypoints2);
		surf.downloadDescriptors(descriptors1GPU, descriptors1);
		surf.downloadDescriptors(descriptors2GPU, descriptors2);

		//// drawing the results
		//Mat img_matches;
		//drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "*************Surf algho time is: " << elapsed.count() << " s\n";

		double max_dist = 0;
		double min_dist = 100;

		//--Quick calculation of min-max distances between keypoints
		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		printf("-- Max dist : %f \n", max_dist);
		printf("-- Min dist : %f \n", min_dist);

		//--Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		std::vector< Point2f > obj;
		std::vector< Point2f > scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//--Get the keypoints from the good matches
			obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}

		//-- Draw matches
		Mat img_matches2;
		drawMatches(image1, keypoints1, image2, keypoints2, good_matches, img_matches2, Scalar::all(-1),
			Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		

		Mat H = findHomography(obj, scene, RANSAC);
		homo = H;
		first = false;
	}

	
	
	// Use the homography Matrix to warp the images
	cv::cuda::GpuMat result;
	cv::cuda::GpuMat gpuInput = cv::cuda::GpuMat(image1);
	cv::cuda::warpPerspective(gpuInput, result, homo, cv::Size(image1.cols + image2.cols, image1.rows));
	
	//cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	//image2.copyTo(half);

	cv::Mat result1 = Mat(result);

	Mat imgResult(image1.rows, image1.cols + image2.cols, image1.type());

	Mat roiImgResult_Left = imgResult(Rect(0, 0, image1.cols, image1.rows));
	Mat roiImgResult_Right = imgResult(Rect(image1.cols, 0, image1.cols, image2.rows));

	Mat roiImg1 = result1(Rect(image2.cols, 0, image2.cols, image2.rows));
	Mat roiImg2 = image2(Rect(0, 0, image2.cols, image2.rows));

	roiImg2.copyTo(roiImgResult_Left); //Img2 will be on the left of imgResult
	roiImg1.copyTo(roiImgResult_Right); //Img1 will be on the right of imgResult

										/* To remove the black portion after stitching, and confine in a rectangular region*/
	Mat fresult(imgResult);
	// vector with all non-black point positions
	std::vector<cv::Point> nonBlackList;
	nonBlackList.reserve(fresult.rows*fresult.cols);

	// add all non-black points to the vector
	// there are more efficient ways to iterate through the image
	for (int j = 0; j<fresult.rows; ++j)
		for (int i = 0; i<fresult.cols; ++i)
		{
			// if not black: add to the list
			if (fresult.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0))
			{
				nonBlackList.push_back(cv::Point(i, j));
			}
		}


	// create bounding rect around those points
	cv::Rect bb = cv::boundingRect(nonBlackList);
 
	return fresult(bb);

}
int main(int argc, char** argv)
{


	//Load the images
	Mat image1;
	Mat image2;
	VideoCapture capture1("F:\\panaroma\\video\\C0003.mp4");
	VideoCapture capture2("F:\\panaroma\\video\\C0004.mp4");

	while (true) {
		auto start = std::chrono::high_resolution_clock::now();
		capture1 >> image1;
		capture2 >> image2;
		Mat result = paranoma(image1, image2);
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "*************Elapsed time new: " << elapsed.count() << " s\n";
		imshow("res", result);
		int key=waitKey(100);
		if (key == 27) {
			break;
		}

	}



	return 0;
}