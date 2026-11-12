#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <array>
#include <cctype>
#include <filesystem>   // <— add this
#include "features.h"

namespace fs = std::filesystem;

// Try to parse a float safely
static bool try_parse_float(const std::string& s, float& out) {
    char* end = nullptr;
    out = std::strtof(s.c_str(), &end);
    return end && end != s.c_str(); // something got parsed
}

// Read CSV -> vector of rows [ax, ay, az]; picks first 3 numeric fields per line
static std::vector<std::array<float,3>> read_csv(const std::string& path) {
    std::vector<std::array<float,3>> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << path << "\n";
        return data;
    }

    // --- detect delimiter from first line ---
    std::string header;
    if (!std::getline(file, header)) return data;
    char delim = ',';
    if (header.find(';') != std::string::npos && header.find(',') == std::string::npos) delim = ';';

    // split header
    std::vector<std::string> cols;
    {
        std::stringstream ss(header);
        std::string tok;
        while (std::getline(ss, tok, delim)) cols.push_back(tok);
    }

    // find indices for ax, ay, az (case-insensitive, multiple aliases)
    auto lower = [](std::string s){ for(char& c:s) c=std::tolower((unsigned char)c); return s; };
    int i_ax = -1, i_ay = -1, i_az = -1;

    for (int i=0;i<(int)cols.size();++i) {
        std::string c = lower(cols[i]);
        if (i_ax<0 && (c=="ax" || c=="accel_x" || c=="x")) i_ax=i;
        else if (i_ay<0 && (c=="ay" || c=="accel_y" || c=="y")) i_ay=i;
        else if (i_az<0 && (c=="az" || c=="accel_z" || c=="z")) i_az=i;
    }

    // fallback: pick first three *numeric-looking* columns if names not found
    if (i_ax<0 || i_ay<0 || i_az<0) {
        // scan first data line to find numeric columns
        std::string firstDataLine;
        if (!std::getline(file, firstDataLine)) return data;   // no data
        // keep a copy to parse after indices chosen
        std::string firstCopy = firstDataLine;

        std::vector<int> numeric_idx;
        {
            std::stringstream ss(firstDataLine);
            std::string val; int idx=0;
            while (std::getline(ss, val, delim)) {
                char* end=nullptr; std::strtof(val.c_str(), &end);
                if (end && end!=val.c_str()) numeric_idx.push_back(idx);
                ++idx;
            }
        }
        if ((int)numeric_idx.size() >= 3) {
            i_ax = numeric_idx[0]; i_ay = numeric_idx[1]; i_az = numeric_idx[2];
        } else {
            std::cerr << "ERROR: " << path << " — cannot find 3 numeric columns\n";
            return data;
        }

        // parse the saved first data line
        {
            std::stringstream ss(firstCopy);
            std::string val; int idx=0;
            std::array<float,3> row{};
            float f_ax=0,f_ay=0,f_az=0;
            while (std::getline(ss, val, delim)) {
                if (idx==i_ax) f_ax=std::strtof(val.c_str(), nullptr);
                if (idx==i_ay) f_ay=std::strtof(val.c_str(), nullptr);
                if (idx==i_az) f_az=std::strtof(val.c_str(), nullptr);
                ++idx;
            }
            row[0]=f_ax; row[1]=f_ay; row[2]=f_az;
            data.push_back(row);
        }
    }

    // info (helps verify we’re aligned)
    std::cerr << "[CSV] " << path << " -> using columns: "
              << (i_ax>=0?cols[i_ax]:"col"+std::to_string(i_ax)) << ", "
              << (i_ay>=0?cols[i_ay]:"col"+std::to_string(i_ay)) << ", "
              << (i_az>=0?cols[i_az]:"col"+std::to_string(i_az)) << "\n";

    // if we didn’t consume the first data line above, continue normally; otherwise continue from current
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val; int idx=0;
        float f_ax=0,f_ay=0,f_az=0;
        bool ok_ax=false, ok_ay=false, ok_az=false;
        while (std::getline(ss, val, delim)) {
            if (idx==i_ax) { f_ax=std::strtof(val.c_str(), nullptr); ok_ax=true; }
            if (idx==i_ay) { f_ay=std::strtof(val.c_str(), nullptr); ok_ay=true; }
            if (idx==i_az) { f_az=std::strtof(val.c_str(), nullptr); ok_az=true; }
            ++idx;
        }
        if (ok_ax && ok_ay && ok_az) {
            data.push_back({f_ax, f_ay, f_az});
        }
    }
    return data;
}

int main() {
    const std::string ROOT = "../data/new_test_dataset";

    // Collect ALL *.csv files under ROOT/<class>/
    std::vector<std::string> paths;
    for (const auto& class_dir : fs::directory_iterator(ROOT)) {
        if (!class_dir.is_directory()) continue;
        for (const auto& f : fs::directory_iterator(class_dir.path())) {
            if (!f.is_regular_file()) continue;
            if (f.path().extension() == ".csv") {
                paths.push_back(f.path().string());
            }
        }
    }
    if (paths.empty()) {
        std::cerr << "ERROR: No CSV files found under " << ROOT << "\n";
        return 1;
    }

    std::ofstream fout_feat("../logs/features_cpp.csv");
    std::ofstream fout_scaled("../logs/scaled_features_cpp.csv");
    if (!fout_feat.is_open() || !fout_scaled.is_open()) {
        std::cerr << "ERROR: cannot open logs/ outputs\n";
        return 1;
    }

    // headers
    fout_feat   << "jerk_rms,centroid_norm,zcr,crest_factor_amp,ber_f0,high_ratio,bandpower_over_entropy_amp\n";
    fout_scaled << "jerk_rms,centroid_norm,zcr,crest_factor_amp,ber_f0,high_ratio,bandpower_over_entropy_amp\n";

    // Process every CSV
    for (const auto& p : paths) {
        auto data = read_csv(p);
        if (data.size() < WIN_SAMPLES) {
            std::cerr << "SKIP (too short): " << p << "  len=" << data.size() << "\n";
            continue;
        }
        size_t windows_here = 0;
        for (size_t s = 0; s + WIN_SAMPLES <= data.size(); s += 64) {
            float imu[WIN_SAMPLES][IMU_AXES];
            for (int i = 0; i < WIN_SAMPLES; ++i) {
                for (int j = 0; j < IMU_AXES; ++j) {
                    imu[i][j] = data[s + i][j];
                }
            }

            float feats[FEAT_DIM];
            compute_features(imu, feats);

            // raw
            for (int i = 0; i < FEAT_DIM; ++i)
                fout_feat << feats[i] << (i < FEAT_DIM - 1 ? "," : "\n");

            // scaled
            standardize_features(feats);
            for (int i = 0; i < FEAT_DIM; ++i)
                fout_scaled << feats[i] << (i < FEAT_DIM - 1 ? "," : "\n");

            ++windows_here;
        }
        std::cout << p << " -> windows: " << windows_here << "\n";
    }
    std::cout << "✅ C++ features + scaled features saved to logs/\n";
    return 0;
}