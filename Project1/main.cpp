#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// === Параметры доски и камеры ===
const Size boardSize(7, 7);       // количество внутренних углов шахматной доски
const float squareSize = 0.05f;   // размер клетки, м (5 см)
const double pixelSize_mm = 0.0055; // размер пикселя в мм

// === Генерация 3D-точек шахматной доски ===
vector<Point3f> createChessboardCorners(Size boardSize, float squareSize) {
    vector<Point3f> corners;
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
    return corners;
}

// === Калибровка по одному видео ===
void processVideo(const string& videoPath, const string& outputPrefix, int frameStep = 1) {
    VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        cerr << "Не удалось открыть видео: " << videoPath << endl;
        return;
    }

    // Подготовка для записи видео
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter writer("output/corners_" + outputPrefix + ".avi",
        VideoWriter::fourcc('M', 'J', 'P', 'G'), 20,
        Size(frame_width, frame_height));

    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    vector<Point3f> objp = createChessboardCorners(boardSize, squareSize);

    int frameCount = 0;
    Mat frame, gray;
    //int maxFrames = 30; // обработаем только 100 кадров

    while (true) {
        cap >> frame;
        //if (frame.empty() || frameCount >= maxFrames) break;
        if (frame.empty()) break;
        frameCount++;
        //cout << "Кадр номер(" << frameCount << "): " << endl;


        //double elapsed = (cv::getTickCount() - start) / cv::getTickFrequency();
        //if (elapsed > maxTime) break;

        // Пропускаем кадры для прореживания
        if (frameCount % frameStep != 0) continue;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, boardSize, corners,
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001));
            drawChessboardCorners(frame, boardSize, corners, found);
            objectPoints.push_back(objp);
            imagePoints.push_back(corners);
        }

        writer.write(frame);

        // Показываем текущий кадр в окне:
        //imshow("Calibration process", frame);
        //if (waitKey(10) == 27) break; // выход по ESC
    }

    cap.release();
    writer.release();

    // === Калибровка ===
    if (objectPoints.size() < 5) {
        cerr << "Недостаточно кадров с доской для калибровки: " << objectPoints.size() << endl;
        return;
    }

    Mat cameraMatrix, distCoeffs, rvecs, tvecs;
    vector<Mat> rvecsOut, tvecsOut;
    double rms = calibrateCamera(objectPoints, imagePoints, Size(frame_width, frame_height),
        cameraMatrix, distCoeffs, rvecsOut, tvecsOut);

    cout << "RMS ошибка калибровки (" << outputPrefix << "): " << rms << endl;

    // === Сохранение результатов ===
    ofstream fout("output/calibration_results_" + outputPrefix + ".txt");
    fout << "RMS = " << rms << "\n";
    fout << "Camera Matrix:\n" << cameraMatrix << "\n";
    fout << "Distortion Coefficients:\n" << distCoeffs << "\n";

    // Фокусное расстояние в мм
    double fx_mm = cameraMatrix.at<double>(0, 0) * pixelSize_mm;
    double fy_mm = cameraMatrix.at<double>(1, 1) * pixelSize_mm;
    fout << "Focal length (mm): fx = " << fx_mm << ", fy = " << fy_mm << "\n";
    fout.close();

    // === Сохранение калибровочных точек ===
    ofstream pointsFile("output/calibration_points.txt");
    for (size_t i = 0; i < imagePoints.size(); ++i) {
        pointsFile << "Frame " << i << ":\n";
        for (const auto& p : imagePoints[i])
            pointsFile << p.x << " " << p.y << "\n";
        pointsFile << "\n";
    }
    pointsFile.close();
}

int main() {
    std::setlocale(LC_ALL, "");
    filesystem::create_directory("output");

    // Основные видео
    cout << "=== Процесс первого видео запущен ===" << endl;
    processVideo("C:/Users/bugro/Videos/Calibration_01.mp4", "Calibration_01");
    cout << "=== Процесс второго видео запущен ===" << endl;
    processVideo("C:/Users/bugro/Videos/Calibration_02.mp4", "Calibration_02");

    // Прореживание кадров (в 2, 4, 8 раз)
    cout << "=== Процесс первое видео прореживание в 2 раза ===" << endl;
    processVideo("C:/Users/bugro/Videos/Calibration_01.mp4", "Calibration_01_downsample_2", 2);
    cout << "=== Процесс первое видео прореживание в 4 раза  ===" << endl;
    processVideo("C:/Users/bugro/Videos/Calibration_01.mp4", "Calibration_01_downsample_4", 4);
    cout << "=== Процесс первое видео прореживание в 8 раз  ===" << endl;
    processVideo("C:/Users/bugro/Videos/Calibration_01.mp4", "Calibration_01_downsample_8", 8);

    cout << "=== Калибровка завершена ===" << endl;
    return 0;
}