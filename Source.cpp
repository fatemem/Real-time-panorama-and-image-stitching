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



Mat paranoma(Mat image1,Mat image2) {
	Size size(1024, 780);

	cv::resize(image1, image1, size);
	cv::resize(image2, image2, size);

	Mat gray_image1;
	Mat gray_image2;

	//Covert to Grayscale
	cvtColor(image1, gray_image1, cv::COLOR_BGRA2GRAY);
	cvtColor(image2, gray_image2, cv::COLOR_BGRA2GRAY);


	

	//--Step 1 : Detect the keypoints using SURF Detector




	//--Step 1 : Detect the keypoints using SURF Detector
	auto start = std::chrono::high_resolution_clock::now();
	SURF_CUDA surf = cv::cuda::SURF_CUDA::SURF_CUDA(400, 4, 3, false, 0.01f, false);
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

	

	//Find the Homography Matrix

	Mat H = findHomography(obj, scene, RANSAC);
	
	if (H.empty()) {
		return H;
	}
	
	// Use the homography Matrix to warp the images
	cv::cuda::GpuMat gpuOutput;
	cv::cuda::GpuMat gpuInput = cv::cuda::GpuMat(image1);

	cv::cuda::warpPerspective(gpuInput, gpuOutput, H, cv::Size(image1.cols + image2.cols, image1.rows), INTER_LINEAR, BORDER_CONSTANT, 0, Stream::Null());
	cv::Mat result = Mat(gpuOutput);
	cv::Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);
	/* To remove the black portion after stitching, and confine in a rectangular region*/

	// vector with all non-black point positions
	std::vector<cv::Point> nonBlackList;
	nonBlackList.reserve(result.rows*result.cols);

	// add all non-black points to the vector
	// there are more efficient ways to iterate through the image
	for (int j = 0; j<result.rows; ++j)
		for (int i = 0; i<result.cols; ++i)
		{
			// if not black: add to the list
			if (result.at<cv::Vec3b>(j, i) != cv::Vec3b(0, 0, 0))
			{
				nonBlackList.push_back(cv::Point(i, j));
			}
		}


	// create bounding rect around those points
	cv::Rect bb = cv::boundingRect(nonBlackList);
	
	//return result(bb);
	return result;


}
int main(int argc, char** argv)
{


	//Load the images
	Mat image1;
	Mat image2;
	VideoCapture capture1("F:\\video\\C0003.mp4");
	VideoCapture capture2("F:\\video\\C0004.mp4");

	while (true) {
		auto start = std::chrono::high_resolution_clock::now();
		capture1 >> image1;
		capture2 >> image2;
		Mat result = paranoma(image1, image2);
		if (result.empty()) {
			continue;
		}
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "*************Elapsed time new: " << elapsed.count() << " s\n";
		imshow("res", result);
		waitKey(100);

	}

	

	return 0;
}