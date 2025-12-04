#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace cv;

// Функция для обработки видеофайла
void processVideo(const string& inputPath, const string& outputPathSegments, const string& outputPathContours, const string& timeLogPath) {
    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        cerr << "Ошибка: не удалось открыть видеофайл: " << inputPath << endl;
        return;
    }

    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));
    Size frameSize(static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));

    VideoWriter writerSegments(outputPathSegments, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
    VideoWriter writerContours(outputPathContours, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);

    if (!writerSegments.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для сегментов: " << outputPathSegments << endl;
        return;
    }
    if (!writerContours.isOpened()) {
        cerr << "Ошибка: не удалось открыть выходной видеофайл для контуров: " << outputPathContours << endl;
        return;
    }

    ofstream timeLogFile(timeLogPath);
    if (!timeLogFile.is_open()) {
        cerr << "Ошибка: не удалось открыть файл для лога времени: " << timeLogPath << endl;
        return;
    }

    Mat frame, segmented, gray, edges, contoursOverlay;
    int frameCount = 0;

    vector<double> processingTimes;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        auto start = chrono::high_resolution_clock::now();

        // 1. Применить MeanShift
        pyrMeanShiftFiltering(frame, segmented, 20, 30, 2);

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0; // в мс
        processingTimes.push_back(duration);
        timeLogFile << duration << endl;

        // 3. Преобразовать в оттенки серого и применить Canny
        cvtColor(segmented, gray, COLOR_BGR2GRAY);
        Canny(gray, edges, 50, 150);

        // 4. Наложить контуры на сегментированное изображение
        contoursOverlay = segmented.clone(); // Копируем сегментированное изображение

        // Вариант 1: Использовать findContours/drawContours
        //vector<vector<Point>> contours;
        //vector<Vec4i> hierarchy;
        //findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        //drawContours(contoursOverlay, contours, -1, Scalar(0, 255, 0), 1, LINE_8); // Зеленые контуры

        // Вариант 2: Попиксельная обработка
        
        for (int y = 0; y < edges.rows; ++y) {
            for (int x = 0; x < edges.cols; ++x) {
                if (edges.at<uchar>(y, x) != 0) { // Если пиксель является частью контура
                    contoursOverlay.at<Vec3b>(y, x) = Vec3b(0, 255, 0); // Установить зеленый цвет
                }
            }
        }
        

        // 2. Записать сегментированное изображение
        writerSegments.write(segmented);

        // 5. Записать изображение с наложенными контурами
        writerContours.write(contoursOverlay);

        frameCount++;
        if (frameCount % 30 == 0) { // Прогресс каждые 30 кадров
            cout << "Обработано кадров: " << frameCount << " из " << totalFrames << endl;
        }
    }

    cap.release();
    writerSegments.release();
    writerContours.release();
    timeLogFile.close();

    cout << "Видео " << inputPath << " обработано." << endl;
    cout << "Выходные файлы: " << outputPathSegments << ", " << outputPathContours << endl;
    cout << "Лог времени: " << timeLogPath << endl;

    // Вывод статистики по времени
    if (!processingTimes.empty()) {
        double sum = 0, sumSquares = 0;
        double minTime = processingTimes[0], maxTime = processingTimes[0];
        for (double t : processingTimes) {
            sum += t;
            sumSquares += t * t;
            if (t < minTime) minTime = t;
            if (t > maxTime) maxTime = t;
        }
        double mean = sum / processingTimes.size();
        double variance = (sumSquares / processingTimes.size()) - (mean * mean);
        double stddev = sqrt(variance);

        cout << "--- Статистика по времени обработки (мс) для " << inputPath << " ---" << endl;
        cout << "Среднее: " << mean << endl;
        cout << "Минимум: " << minTime << endl;
        cout << "Максимум: " << maxTime << endl;
        cout << "Стандартное отклонение: " << stddev << endl;
    }
}

int main(int argc, char* argv[]) {
    // Обработка первого видеофайла
    processVideo("C:/Users/bugro/Videos/city.mp4",
        "city_segments.avi",
        "city_contours.avi",
        "city_timing_log.txt");

    // Обработка второго видеофайла
    processVideo("C:/Users/bugro/Videos/winter.mp4",
        "winter_segments.avi",
        "winter_contours.avi",
        "winter_timing_log.txt");

    cout << "Обработка завершена. Проверьте выходные файлы." << endl;

    return 0;
}