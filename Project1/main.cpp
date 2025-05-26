#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

// Настройки
#define DRAW_MATCHES true
#define DRAW_KEYPOINTS_MODE 1
#define MIN_MATCH_DISTANCE 100.0
#define LOWE_RATIO 0.7

// Цвета для визуализации
const cv::Scalar MATCHED_COLOR(0, 0, 255);    // Красный для сопоставленных
const cv::Scalar UNMATCHED_COLOR(0, 255, 0);  // Зеленый для несопоставленных
const cv::Scalar LINE_COLOR(255, 0, 0);       // Синий для линий сопоставлений


int main(int argc, char** argv) {
    cv::namedWindow("SIFT Tracking", cv::WINDOW_AUTOSIZE);

    cv::VideoCapture cap("C:/Users/bugro/Videos/Grunt.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Ошибка открытия видео!" << std::endl;
        return -1;
    }

    // Получите параметры видео из исходного файла
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size frameSize(
        (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );

    // Создайте VideoWriter (кодек: MP4V, путь: output.mp4)
    cv::VideoWriter writer("output.mp4",
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps,
        frameSize);

    std::ofstream statsFile("sift_stats.txt");
    statsFile << "Frame,TotalFeatures,MatchedFeatures,DetectTime(ms),MatchTime(ms)\n";

    auto detector = cv::SIFT::create();
    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    cv::Mat prevGray;
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;

    int frameNum = 0;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        auto startTime = std::chrono::steady_clock::now();

        // Обработка изображения
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        auto detectStart = std::chrono::steady_clock::now();
        detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        auto detectEnd = std::chrono::steady_clock::now();

        // Сопоставление признаков
        std::vector<std::vector<cv::DMatch>> knnMatches;
        std::vector<cv::DMatch> goodMatches;
        double matchTime = 0;

        if (!prevKeypoints.empty()) {
            auto matchStart = std::chrono::steady_clock::now();
            matcher->knnMatch(descriptors, prevDescriptors, knnMatches, 2);
            matchTime = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - matchStart).count();

            for (auto& match : knnMatches) {
                if (match[0].distance < LOWE_RATIO * match[1].distance &&
                    match[0].distance < MIN_MATCH_DISTANCE) {
                    goodMatches.push_back(match[0]);
                }
            }
        }

        // Визуализация
        cv::Mat outputFrame;
        cv::cvtColor(gray, outputFrame, cv::COLOR_GRAY2BGR);

        // Разделение точек на сопоставленные/несопоставленные
        std::vector<bool> isMatched(keypoints.size(), false);
        std::vector<cv::KeyPoint> matchedKeypoints, unmatchedKeypoints;

        for (auto& m : goodMatches) {
            isMatched[m.queryIdx] = true;
            matchedKeypoints.push_back(keypoints[m.queryIdx]);
        }

        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (!isMatched[i]) {
                unmatchedKeypoints.push_back(keypoints[i]);
            }
        }

        // Рисуем несопоставленные точки
        cv::drawKeypoints(outputFrame, unmatchedKeypoints, outputFrame,
            UNMATCHED_COLOR, DRAW_KEYPOINTS_MODE ? cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
            : cv::DrawMatchesFlags::DEFAULT);

        // Рисуем сопоставленные точки
        cv::drawKeypoints(outputFrame, matchedKeypoints, outputFrame,
            MATCHED_COLOR, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // Рисуем линии сопоставлений
        if (!prevKeypoints.empty() && DRAW_MATCHES) {
            for (auto& m : goodMatches) {
                cv::line(outputFrame,
                    keypoints[m.queryIdx].pt,
                    prevKeypoints[m.trainIdx].pt,
                    LINE_COLOR);
            }
        }

        // Обновление данных
        prevGray = gray.clone();
        prevKeypoints = keypoints;
        prevDescriptors = descriptors;

        // Запись статистики
        statsFile << frameNum << ","
            << keypoints.size() << ","
            << goodMatches.size() << ","
            << std::chrono::duration<double, std::milli>(detectEnd - detectStart).count() << ","
            << matchTime << "\n";

        // Отображение
        cv::imshow("SIFT Tracking", outputFrame);
        writer.write(outputFrame);
        if (cv::waitKey(30) >= 0) break;

        frameNum++;
    }

    cap.release();
    writer.release(); // Важно!
    statsFile.close();
    cv::destroyAllWindows();
    return 0;
}