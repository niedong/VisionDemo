// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine Human Pose Estimation demo application
* \file human_pose_estimation_demo/main.cpp
* \example human_pose_estimation_demo/main.cpp
*/

#include <vector>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "human_pose_estimation_demo.hpp"
#include "human_pose_estimator.hpp"
#include "render_human_pose.hpp"

#define w 1200
#define h 800

using namespace cv;
using namespace InferenceEngine;
using namespace human_pose_estimation;

int flag = -1;
Point2f head(-1.0, -1.0);
Point2f tail(-1.0, -1.0);
Point2f v(-1.0, -1.0);

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
	// ---------------------------Parsing and validation of input args--------------------------------------

	gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
	if (FLAGS_h)
	{
		showUsage();
		return false;
	}

	std::cout << "[ INFO ] Parsing input parameters" << std::endl;

	if (FLAGS_i.empty())
	{
		throw std::logic_error("Parameter -i is not set");
	}

	if (FLAGS_m.empty())
	{
		throw std::logic_error("Parameter -m is not set");
	}

	return true;
}

void refresh(cv::Mat &img)
{
	rectangle(img, Point(0, 0), Point(w, h), Scalar(0, 0, 0), -1);
}

void drawHuman(const std::vector<HumanPose> &poses, cv::Mat &img)
{
	const std::vector<std::pair<int, int>> limbKeypointsIds = {
		{1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13}, {1, 0}};

	const Point2f absentKeypoint(-1.0f, -1.0f);

	/*	
	for (const auto &pose : poses) {
		CV_Assert(pose.keypoints.size() == HumanPoseEstimator::keypointsNumber);
		for (size_t i = 0; i < pose.keypoints.size(); i++) {
			if (pose.keypoints[i] != absentKeypoint) {
				circle(img, pose.keypoints[i], 3, Scalar(255, 255, 255), -1);
			}
		}
	}
	*/

	for (const auto &pose : poses)
	{
		for (const auto &limbKeypointsId : limbKeypointsIds)
		{
			std::pair<cv::Point2f, cv::Point2f> limbKeypoints(
				pose.keypoints[limbKeypointsId.first], pose.keypoints[limbKeypointsId.second]);
			if (limbKeypoints.first == absentKeypoint || limbKeypoints.second == absentKeypoint)
			{
				continue;
			}

			float meanX = (limbKeypoints.first.x + limbKeypoints.second.x) / 2;
			float meanY = (limbKeypoints.first.y + limbKeypoints.second.y) / 2;
			cv::Point difference = limbKeypoints.first - limbKeypoints.second;
			double length = std::sqrt(difference.x * difference.x + difference.y * difference.y);
			int angle = static_cast<int>(std::atan2(difference.y, difference.x) * 180 / CV_PI);
			std::vector<cv::Point> polygon;
			ellipse2Poly(Point2d(meanX, meanY), Size2d(length / 2, 4), angle, 0, 360, 1, polygon);
			fillConvexPoly(img, polygon, Scalar(255, 255, 255));
		}
	}

	for (const auto &pose : poses)
	{
		if (pose.keypoints[0] == absentKeypoint || pose.keypoints[1] == absentKeypoint)
			continue;

		float X1 = pose.keypoints[0].x;
		float Y1 = pose.keypoints[0].y;
		float X2 = pose.keypoints[1].x;
		float Y2 = pose.keypoints[1].y;

		float meanX = (X1 + X2) / 2;
		float meanY = (Y1 + Y2) / 2;

		Point diff(X1 - X2, Y1 - Y2);
		double length = std::sqrt(diff.x * diff.x + diff.y * diff.y);
		int angle = static_cast<int>(std::atan2(diff.y, diff.x) * 180 / CV_PI);
		std::vector<Point> polygon;
		ellipse2Poly(Point2d(meanX, meanY), Size2d(length / 2, length / 3), angle, 0, 360, 1, polygon);
		fillConvexPoly(img, polygon, Scalar(255, 255, 255));
	}

	for (const auto &pose : poses)
	{
		if (pose.keypoints[1] == absentKeypoint)
			continue;
		if (pose.keypoints[8] == absentKeypoint && pose.keypoints[11] == absentKeypoint)
			continue;

		float X1 = pose.keypoints[1].x;
		float Y1 = pose.keypoints[1].y;
		float X2 = 0.0, Y2 = 0.0;
		if (pose.keypoints[8] == absentKeypoint)
		{
			X2 = pose.keypoints[11].x;
			Y2 = pose.keypoints[11].y;
		}
		else if (pose.keypoints[11] == absentKeypoint)
		{
			X2 = pose.keypoints[8].x;
			Y2 = pose.keypoints[8].y;
		}
		else
		{
			X2 = (pose.keypoints[8].x + pose.keypoints[11].x) / 2;
			Y2 = (pose.keypoints[8].y + pose.keypoints[11].y) / 2;
		}

		Point diff(X1 - X2, Y1 - Y2);
		double length = std::sqrt(diff.x * diff.x + diff.y * diff.y);

		float X = 0.0, Y = 0.0;
		X = X1 - ((Y1 - Y2) / (float)length) * (float)length / 4;
		Y = Y1 + ((X1 - X2) / (float)length) * (float)length / 4;
		Point2f p1(X, Y);

		X = X1 - X + X1;
		Y = Y1 - Y + Y1;
		Point2f p2(X, Y);

		X = X2 - ((Y1 - Y2) / (float)length) * (float)length / 4;
		Y = Y2 + ((X1 - X2) / (float)length) * (float)length / 4;
		Point2f p3(X, Y);

		X = X2 - X + X2;
		Y = Y2 - Y + Y2;
		Point2f p4(X, Y);

		std::vector<Point> cont;
		cont.reserve(4);
		cont.push_back(p1);
		cont.push_back(p2);
		cont.push_back(p4);
		cont.push_back(p3);
		std::vector<std::vector<Point>> conts;
		conts.push_back(cont);

		polylines(img, conts, true, Scalar(255, 255, 255), 2, LINE_AA);
		fillPoly(img, conts, Scalar(255, 255, 255));
	}
}

void drawBow(const std::vector<HumanPose> &poses, cv::Mat &img)
{
	const Point2f absentKeypoint(-1.0f, -1.0f);
	for (const auto &pose : poses)
	{
		if (pose.keypoints[6] == absentKeypoint || pose.keypoints[7] == absentKeypoint)
			continue;

		float X2 = pose.keypoints[6].x;
		float Y2 = pose.keypoints[6].y;
		float X1 = pose.keypoints[7].x;
		float Y1 = pose.keypoints[7].y;

		Point diff(X1 - X2, Y1 - Y2);
		double length = std::sqrt(diff.x * diff.x + diff.y * diff.y);

		float X3 = X1 - ((Y1 - Y2) / (float)length) * (float)length / 2;
		float Y3 = Y1 + ((X1 - X2) / (float)length) * (float)length / 2;
		Point2f p1(X3, Y3);

		Point p3(X3 - diff.x, Y3 - diff.y);

		float X4 = X1 - X3 + X1;
		float Y4 = Y1 - Y3 + Y1;
		Point2f p2(X4, Y4);

		Point p4(X4 - diff.x, Y4 - diff.y);

		diff.x = X4 - X3;
		diff.y = Y4 - Y3;

		Point2f p5(p3.x - diff.x, p3.y - diff.y);
		Point2f p6(p4.x + diff.x, p4.y + diff.y);

		line(img, p1, p2, Scalar(255, 255, 255), 2);
		line(img, p1, p5, Scalar(255, 255, 255), 2);
		line(img, p2, p6, Scalar(255, 255, 255), 2);

		if (pose.keypoints[4] == absentKeypoint)
			continue;

		Point2f p7 = pose.keypoints[4];
		line(img, p5, p7, Scalar(255, 255, 255), 1);
		line(img, p6, p7, Scalar(255, 255, 255), 1);
	}
}

void drawArrow(cv::Mat &img)
{
	head.x += v.x;
	head.y += v.y;

	tail.x += v.x;
	tail.y += v.y;
	line(img, head, tail, Scalar(255, 255, 255), 2);

	if (head.x >= w || head.y >= h || head.x <= 0 || head.y <= 0)
		flag = -1;
}

void check(const std::vector<HumanPose> &poses)
{
	const Point2f absentKeypoint(-1.0f, -1.0f);
	for (const auto &pose : poses)
	{
		if (pose.keypoints[4] == absentKeypoint || pose.keypoints[7] == absentKeypoint ||
			pose.keypoints[6] == absentKeypoint || pose.keypoints[1] == absentKeypoint)
			continue;

		Point diff(pose.keypoints[4].x - pose.keypoints[7].x,
				   pose.keypoints[4].y - pose.keypoints[7].y);
		double dis = std::sqrt(diff.x * diff.x + diff.y * diff.y);
		if (dis < 20)
		{
			flag = 0;
		}

		Point d1(pose.keypoints[1].x - pose.keypoints[7].x,
				 pose.keypoints[1].y - pose.keypoints[7].y);
		double l1 = std::sqrt(d1.x * d1.x + d1.y * d1.y);
		Point d2(pose.keypoints[4].x - pose.keypoints[7].x,
				 pose.keypoints[4].y - pose.keypoints[7].y);
		double l2 = std::sqrt(d2.x * d2.x + d2.y * d2.y);
		if (l2 > l1 && flag == 0)
		{
			head = pose.keypoints[7];
			tail = pose.keypoints[6];
			v.x = (pose.keypoints[7].x - pose.keypoints[6].x) / 2;
			v.y = (pose.keypoints[7].y - pose.keypoints[6].y) / 2;
			flag = 1;
		}
	}
}

int main(int argc, char *argv[])
{
	try
	{
		std::cout << "InferenceEngine: " << GetInferenceEngineVersion() << std::endl;

		// ------------------------------ Parsing and validation of input args ---------------------------------
		if (!ParseAndCheckCommandLine(argc, argv))
		{
			return EXIT_SUCCESS;
		}

		HumanPoseEstimator estimator(FLAGS_m, FLAGS_d, FLAGS_pc);
		cv::VideoCapture cap;
		if (!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i)))
		{
			throw std::logic_error("Cannot open input file or camera: " + FLAGS_i);
		}

		int delay = 33;
		double inferenceTime = 0.0;
		cv::Mat image;
		if (!cap.read(image))
		{
			throw std::logic_error("Failed to get frame from cv::VideoCapture");
		}
		estimator.estimate(image); // Do not measure network reshape, if it happened
		if (!FLAGS_no_show)
		{
			std::cout << "To close the application, press 'CTRL+C' or any key with focus on the output window" << std::endl;
		}

		std::cout << "*******GUI WINDOW*******" << std::endl;
		Mat img = Mat::zeros(Size(w, h), CV_8UC3);

		do
		{
			double t1 = static_cast<double>(cv::getTickCount());
			std::vector<HumanPose> poses = estimator.estimate(image);
			double t2 = static_cast<double>(cv::getTickCount());
			if (inferenceTime == 0)
			{
				inferenceTime = (t2 - t1) / cv::getTickFrequency() * 1000;
			}
			else
			{
				inferenceTime = inferenceTime * 0.95 + 0.05 * (t2 - t1) / cv::getTickFrequency() * 1000;
			}
			if (FLAGS_r)
			{
				for (HumanPose const &pose : poses)
				{
					std::stringstream rawPose;
					rawPose << std::fixed << std::setprecision(0);
					for (auto const &keypoint : pose.keypoints)
					{
						rawPose << keypoint.x << "," << keypoint.y << " ";
					}
					rawPose << pose.score;
					std::cout << rawPose.str() << std::endl;
				}
			}

			if (FLAGS_no_show)
			{
				continue;
			}

			refresh(img);

			drawHuman(poses, img);

			check(poses);

			if (flag == 0)
			{
				drawBow(poses, img);
			}
			else if (flag == 1)
			{
				drawArrow(img);
			}

			imshow("GUI", img);

			renderHumanPose(poses, image);

			cv::Mat fpsPane(35, 155, CV_8UC3);
			fpsPane.setTo(cv::Scalar(153, 119, 76));
			cv::Mat srcRegion = image(cv::Rect(8, 8, fpsPane.cols, fpsPane.rows));
			cv::addWeighted(srcRegion, 0.4, fpsPane, 0.6, 0, srcRegion);
			std::stringstream fpsSs;
			fpsSs << "FPS: " << int(1000.0f / inferenceTime * 100) / 100.0f;
			cv::putText(image, fpsSs.str(), cv::Point(16, 32),
						cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 255));
			cv::imshow("ICV Human Pose Estimation", image);

			int key = cv::waitKey(delay) & 255;
			if (key == 'p')
			{
				delay = (delay == 0) ? 33 : 0;
			}
			else if (key == 27)
			{
				break;
			}
		} while (cap.read(image));
	}
	catch (const std::exception &error)
	{
		std::cerr << "[ ERROR ] " << error.what() << std::endl;
		return EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "[ ERROR ] Unknown/internal exception happened." << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "[ INFO ] Execution successful" << std::endl;
	return EXIT_SUCCESS;
}
