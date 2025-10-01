#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <map>
#include <algorithm>
#include <set>
#include <iomanip>

// ---------- Настройки -------------
const int MAX_FEATURES = 1000;          // всего признаков на кадр (по умолчанию 1000)
const int GRID_ROWS = 2;                // сетка 2x2
const int GRID_COLS = 2;
const bool DRAW_RICH_KEYPOINTS = false;
const float LOWE_RATIO = 0.7f;
const int KNN_K = 2;
const int MAX_TRACK_HISTORY = 3;     // сколько точек хранить в треке (по желанию)
const double MIN_MATCH_DISTANCE = 10.0; // порог расстояния для ORB matches (можно подстроить)
// -----------------------------------

// Структура трека
struct Track {
    int id;
    std::vector<cv::Point2f> pts;  // история точек (по кадрам, последние в конце)
    cv::Scalar color;
    int lastSeenFrame = 0;
};

// Вспомогательная генерация цвета
cv::Scalar randomColor(int seed) {
    static std::mt19937 rng(12345);
    rng.seed(seed);
    std::uniform_int_distribution<int> u(50, 255);
    return cv::Scalar(u(rng), u(rng), u(rng));
}

int main(int argc, char** argv) {
    std::string inputPath = "C:/Users/bugro/Videos/opencv/Movement 01 (re).mp4";
    std::string outputVideo = "output_tracked.mp4";

    if (argc >= 2) inputPath = argv[1];
    if (argc >= 3) outputVideo = argv[2];

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть видео: " << inputPath << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::Size frameSize(w, h);

    cv::VideoWriter writer(outputVideo,
        cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
        fps, frameSize);

    // ORB detector
    int featuresPerCell = MAX_FEATURES / (GRID_ROWS * GRID_COLS);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_FEATURES); // общий лимит
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);

    std::ofstream statsFile("feature_stats.csv");
    statsFile << "Frame,TotalDetected,TotalMatched,FrameTime_ms,DetectTime_ms,MatchTime_ms\n";

    std::ofstream perCellFile("per_cell_stats.csv");
    // заголовок: Frame, Cell0_Det,Cell1_Det..., Cell0_Matched,Cell1_Matched..., Cell0_Time_ms,Cell1_Time_ms...
    perCellFile << "Frame";
    for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) perCellFile << ",Cell" << i << "_Detected";
    for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) perCellFile << ",Cell" << i << "_Matched";
    for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) perCellFile << ",Cell" << i << "_ProcTime_ms";
    perCellFile << "\n";

    // Треки
    std::map<int, Track> tracks; // id -> track
    std::vector<int> prevTrackIdPerKeypoint; // индекс prev keypoint -> track id
    std::vector<cv::KeyPoint> prevKeypoints;
    cv::Mat prevDescriptors;

    // --- НОВОЕ: для фильтрации матчей по ячейкам ---
    std::vector<int> prevCellIdxPerKeypoint; // индекс ячейки для каждого prevKeypoint

    int frameNum = 0;
    int nextTrackId = 1;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto frameStart = std::chrono::high_resolution_clock::now();

        cv::Mat gray;
        if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else gray = frame.clone();

        // MASK: если в верхней части кадра чёрная замаскированная область, хорошо бы исключить её.
        // Попробуем динамически определить верхнюю строку полностью чёрных пикселей,
        // но при неудаче используем полный кадр.
        int roi_y0 = 0;
        // простая эвристика: найдем первый ряд сверху, где средняя яркость > 5
        for (int y = 0; y < gray.rows / 3; ++y) {
            cv::Scalar meanRow = cv::mean(gray.row(y));
            if (meanRow[0] > 5) { roi_y0 = y; break; }
        }
        // создаём ROI область (от roi_y0 до bottom)
        cv::Rect wholeROI(0, roi_y0, gray.cols, gray.rows - roi_y0);

        // --- детекция по ячейкам сетки, измерение времени детекции по каждой ячейке
        std::vector<cv::KeyPoint> keypoints; keypoints.reserve(MAX_FEATURES);
        std::vector<double> detectTimePerCell(GRID_ROWS * GRID_COLS, 0.0);
        int detectedPerCellCountTotal = 0;

        for (int r = 0; r < GRID_ROWS; ++r) {
            for (int c = 0; c < GRID_COLS; ++c) {
                int cellIdx = r * GRID_COLS + c;

                int x0 = wholeROI.x + c * (wholeROI.width / GRID_COLS);
                int y0 = wholeROI.y + r * (wholeROI.height / GRID_ROWS);
                int cw = (c == GRID_COLS - 1) ? (wholeROI.x + wholeROI.width - x0) : (wholeROI.width / GRID_COLS);
                int ch = (r == GRID_ROWS - 1) ? (wholeROI.y + wholeROI.height - y0) : (wholeROI.height / GRID_ROWS);
                cv::Rect cellRect(x0, y0, cw, ch);
                if (cellRect.width <= 0 || cellRect.height <= 0) continue;

                // Маска для одной ячейки
                cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
                mask(cellRect).setTo(255);

                // Время детекции ячейки
                auto t0 = std::chrono::high_resolution_clock::now();
                std::vector<cv::KeyPoint> kp_cell;
                orb->detect(gray, kp_cell, mask);
                auto t1 = std::chrono::high_resolution_clock::now();
                double t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                detectTimePerCell[cellIdx] += t_ms;

                // Отсортируем kp_cell по отклику и оставим не больше featuresPerCell
                if (!kp_cell.empty()) {
                    std::sort(kp_cell.begin(), kp_cell.end(),
                        [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
                    if ((int)kp_cell.size() > featuresPerCell) kp_cell.resize(featuresPerCell);
                    // добавляем в общий набор
                    for (auto& k : kp_cell) keypoints.push_back(k);
                    detectedPerCellCountTotal += (int)kp_cell.size();
                }
            }
        } // конец по ячейкам

        // Если обнаружено больше, чем MAX_FEATURES (возможна округлость), оставим топ MAX_FEATURES по response
        if ((int)keypoints.size() > MAX_FEATURES) {
            std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response > b.response;
                });
            keypoints.resize(MAX_FEATURES);
        }

        // Вычислим дескрипторы для всех ключевых точек
        cv::Mat descriptors;
        auto detectEnd = std::chrono::high_resolution_clock::now();
        auto detectTimeTotal = std::chrono::duration<double, std::milli>(detectEnd - frameStart).count(); // временно; скорректируем ниже

        if (!keypoints.empty()) {
            auto tdesc0 = std::chrono::high_resolution_clock::now();
            orb->compute(gray, keypoints, descriptors);
            auto tdesc1 = std::chrono::high_resolution_clock::now();
            double computeDescMs = std::chrono::duration<double, std::milli>(tdesc1 - tdesc0).count();
            // корректируем detectTimePerCell суммированием доли времени (простая аппроксимация)
            if (detectedPerCellCountTotal > 0) {
                for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) {
                    // распределим computeDescMs пропорционально обнаруженным признакам в ячейках
                    // (можно заполнить, если нужно более точное измерение)
                }
            }
        }

        // Чтобы узнать, к каким ячейкам относятся обнаруженные keypoints,
        // создадим массив cellIdxPerKeypoint
        std::vector<int> cellIdxPerKeypoint(keypoints.size(), -1);
        for (size_t i = 0; i < keypoints.size(); ++i) {
            int cx = (int)keypoints[i].pt.x;
            int cy = (int)keypoints[i].pt.y;
            int col = std::min(std::max((cx - wholeROI.x) * GRID_COLS / std::max(1, wholeROI.width), 0), GRID_COLS - 1);
            int row = std::min(std::max((cy - wholeROI.y) * GRID_ROWS / std::max(1, wholeROI.height), 0), GRID_ROWS - 1);
            int idx = row * GRID_COLS + col;
            cellIdxPerKeypoint[i] = idx;
        }

        // --- матчинг текущих дескрипторов к предыдущим
        std::vector<cv::DMatch> goodMatches; // current.queryIdx -> prev.trainIdx encoded DMatch(query=currentIdx, train=prevIdx)
        double matchTimeMs = 0.0;
        if (!prevKeypoints.empty() && !prevDescriptors.empty() && !descriptors.empty()) {
            auto matchStart = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<cv::DMatch>> knn;
            try {
                matcher->knnMatch(descriptors, prevDescriptors, knn, KNN_K);
            }
            catch (cv::Exception& e) {
                std::cerr << "Exception in matcher->knnMatch: " << e.what() << std::endl;
            }
            auto matchEnd = std::chrono::high_resolution_clock::now();
            matchTimeMs = std::chrono::duration<double, std::milli>(matchEnd - matchStart).count();

            // --- ИЗМЕНЕНИЕ: фильтрация матчей по совпадению ячейки ---
            for (size_t i = 0; i < knn.size(); ++i) {
                if (knn[i].empty()) continue;

                bool passesRatio = false;
                if (knn[i].size() >= 2) {
                    passesRatio = (knn[i][0].distance < LOWE_RATIO * knn[i][1].distance);
                }
                else {
                    passesRatio = true;
                }
                if (!passesRatio) continue;
                if (knn[i][0].distance >= MIN_MATCH_DISTANCE) continue;

                int curIdx = (int)i;
                int prevIdx = knn[i][0].trainIdx;
                if (prevIdx < 0 || prevIdx >= (int)prevTrackIdPerKeypoint.size()) continue;
                if (curIdx < 0 || curIdx >= (int)cellIdxPerKeypoint.size()) continue;
                if (prevIdx < 0 || prevIdx >= (int)prevCellIdxPerKeypoint.size()) continue;

                int curCell = cellIdxPerKeypoint[curIdx];
                int prevCell = prevCellIdxPerKeypoint[prevIdx];

                // допускаем матч только внутри одной и той же ячейки
                if (curCell != prevCell) continue;

                // (опционально) можно добавить проверку на физическое смещение:
                // float dx = keypoints[curIdx].pt.x - prevKeypoints[prevIdx].pt.x;
                // float dy = keypoints[curIdx].pt.y - prevKeypoints[prevIdx].pt.y;
                // if (std::sqrt(dx*dx + dy*dy) > SOME_PIXEL_THRESHOLD) continue;

                // принимаем матч
                goodMatches.push_back(knn[i][0]);
            }
        }

        // --- обновление треков: для каждого матча присваиваем текущую точку к треку prevTrackIdPerKeypoint[prevIdx]
        // prevTrackIdPerKeypoint: индекс prev keypoints -> track id (информация сохранилась с предыдущего шага)
        std::vector<int> matchedCurrentIdx(keypoints.size(), 0);
        std::map<int, bool> prevTrackSeenThisFrame; // trackId -> seen?
        std::vector<int> newTrackIdsForKeypoint(keypoints.size(), -1);

        for (auto& m : goodMatches) {
            int curIdx = m.queryIdx;
            int prevIdx = m.trainIdx;
            if (prevIdx >= 0 && prevIdx < (int)prevTrackIdPerKeypoint.size()) {
                int trackId = prevTrackIdPerKeypoint[prevIdx];
                if (trackId > 0 && tracks.find(trackId) != tracks.end()) {
                    // добавляем точку в трек
                    tracks[trackId].pts.push_back(keypoints[curIdx].pt);
                    if ((int)tracks[trackId].pts.size() > MAX_TRACK_HISTORY) tracks[trackId].pts.erase(tracks[trackId].pts.begin());
                    tracks[trackId].lastSeenFrame = frameNum;
                    matchedCurrentIdx[curIdx] = 1;
                    prevTrackSeenThisFrame[trackId] = true;
                    newTrackIdsForKeypoint[curIdx] = trackId;
                }
            }
        }

        // Удалим треки, которые не были сопоставлены в текущем кадре (по условию — если для признака с предыдущего кадра
        // на текущем не обнаружено соответствующего, то этот трек удаляется).
        std::vector<int> toErase;
        for (auto& kv : tracks) {
            int tid = kv.first;
            if (kv.second.lastSeenFrame < frameNum) {
                // не обновлялся в этом кадре
                toErase.push_back(tid);
            }
        }
        for (int tid : toErase) tracks.erase(tid);

        // Для непомеченных текущих ключевых точек — создаём новые треки
        int createdNew = 0;
        for (size_t i = 0; i < keypoints.size(); ++i) {
            if (!matchedCurrentIdx[i]) {
                Track t;
                t.id = nextTrackId++;
                t.pts.push_back(keypoints[i].pt);
                t.color = randomColor(t.id);
                t.lastSeenFrame = frameNum;
                tracks[t.id] = t;
                newTrackIdsForKeypoint[i] = t.id;
                createdNew++;
            }
        }

        // Формируем prevKeypoints/prevDescriptors и mapping prevTrackIdPerKeypoint для следующего кадра.
        prevKeypoints = keypoints;
        prevDescriptors = descriptors.clone();
        prevTrackIdPerKeypoint.resize(prevKeypoints.size());
        for (size_t i = 0; i < prevKeypoints.size(); ++i) {
            prevTrackIdPerKeypoint[i] = newTrackIdsForKeypoint[i];
        }

        // --- НОВОЕ: обновляем prevCellIdxPerKeypoint (ячейка для prevKeypoint)
        prevCellIdxPerKeypoint.resize(prevKeypoints.size());
        for (size_t i = 0; i < prevKeypoints.size(); ++i) {
            // cellIdxPerKeypoint рассчитан для текущего кадра; т.к. prevKeypoints = keypoints, индексы совпадают
            if (i < cellIdxPerKeypoint.size()) prevCellIdxPerKeypoint[i] = cellIdxPerKeypoint[i];
            else prevCellIdxPerKeypoint[i] = -1;
        }

        // Подсчёт количества сопоставленных признаков (unique matched tracks in this frame)
        int totalMatched = 0;
        std::set<int> matchedTrackIds;
        for (auto& m : goodMatches) {
            int prevIdx = m.trainIdx;
            if (prevIdx >= 0 && prevIdx < (int)prevTrackIdPerKeypoint.size()) {
                int tid = prevTrackIdPerKeypoint[prevIdx];
                if (tid > 0) matchedTrackIds.insert(tid);
            }
        }
        totalMatched = (int)matchedTrackIds.size();

        // Подсчёт per-cell detected и matched
        std::vector<int> cellDetected(GRID_ROWS * GRID_COLS, 0);
        std::vector<int> cellMatched(GRID_ROWS * GRID_COLS, 0);
        for (size_t i = 0; i < keypoints.size(); ++i) {
            int ci = cellIdxPerKeypoint[i];
            if (ci >= 0 && ci < (int)cellDetected.size()) cellDetected[ci]++;
            if (newTrackIdsForKeypoint[i] > 0) {
                if (matchedCurrentIdx[i]) cellMatched[ci]++;
            }
        }

        // Общее время обработки кадра (без операций отображения/записи): считаем от frameStart до текущ
        auto frameEnd = std::chrono::high_resolution_clock::now();
        double frameTimeMs = std::chrono::duration<double, std::milli>(frameEnd - frameStart).count();

        // Определим detectTime (we measured per-cell detect times earlier plus descriptor compute time approx)
        double detectTimeMs = 0.0;
        for (double v : detectTimePerCell) detectTimeMs += v;
        // Добавим небольшую аппроксимацию времени compute descriptors (если был)
        double matchTime = matchTimeMs;
        double otherOverhead = frameTimeMs - detectTimeMs - matchTime;
        if (otherOverhead < 0) otherOverhead = 0;

        // Запись общей статистики
        statsFile << frameNum << ","
            << keypoints.size() << ","
            << totalMatched << ","
            << std::fixed << std::setprecision(3) << frameTimeMs << ","
            << detectTimeMs << ","
            << matchTime << "\n";

        // Запись по ячейкам: Frame, Detected x4, Matched x4, ProcTime x4
        perCellFile << frameNum;
        for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) perCellFile << "," << cellDetected[i];
        for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) perCellFile << "," << cellMatched[i];
        int totalDetectedAll = std::max(1, (int)keypoints.size());
        for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) {
            double portion = (double)std::max(0, cellDetected[i]) / (double)totalDetectedAll;
            double cellProc = detectTimePerCell[i] + portion * (matchTime + otherOverhead);
            perCellFile << "," << cellProc;
        }
        perCellFile << "\n";

        // --- Визуализация: рисуем все "живые" треки
        cv::Mat vis;
        if (frame.channels() == 1) cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
        else vis = frame.clone();

        // Рисуем траектории
        for (auto& kv : tracks) {
            const Track& t = kv.second;
            if (t.pts.size() < 1) continue;
            // линии истории
            for (size_t i = 1; i < t.pts.size(); ++i) {
                cv::line(vis, t.pts[i - 1], t.pts[i], t.color, 1, cv::LINE_AA);
            }
            // текущее положение — кружок
            cv::circle(vis, t.pts.back(), 3, t.color, -1);
        }

        // Рисуем сетку
        for (int r = 0; r < GRID_ROWS; r++) {
            int y = wholeROI.y + r * (wholeROI.height / GRID_ROWS);
            cv::line(vis, cv::Point(0, y), cv::Point(w, y), cv::Scalar(200, 200, 200), 1);
        }
        for (int c = 0; c < GRID_COLS; c++) {
            int x = wholeROI.x + c * (wholeROI.width / GRID_COLS);
            cv::line(vis, cv::Point(x, 0), cv::Point(x, h), cv::Scalar(200, 200, 200), 1);
        }

        // Параметры текста
        cv::putText(vis, "Frame: " + std::to_string(frameNum), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        cv::putText(vis, "Detected: " + std::to_string((int)keypoints.size()), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        cv::putText(vis, "Tracked: " + std::to_string(totalMatched), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        writer.write(vis);
        cv::imshow("Feature Tracking", vis);
        if (cv::waitKey(1) == 27) break;

        frameNum++;
    }

    statsFile.close();
    perCellFile.close();
    writer.release();
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Готово. Выходной файл: " << outputVideo << "\n";
    std::cout << "CSV: feature_stats.csv, per_cell_stats.csv\n";
    return 0;
}
