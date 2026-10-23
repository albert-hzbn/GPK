#pragma once
/*
 * input_parser.hpp — INI-style simulation input file parser.
 *
 * File format example (inputs/grain_growth.ini):
 *
 *   # comment
 *   [grid]
 *   Nz    = 1           # 1 → 2-D simulation
 *   Ny    = 64
 *   Nx    = 64
 *   dx    = 1.0e-6      # cell size [m]
 *
 *   [time]
 *   t_end    = 0.14
 *   n_snaps  = 10
 *
 *   [thermo]
 *   M0     = 1.0e-8
 *   sigma0 = 0.5
 *   eps4   = 0.0
 *   D      = 0.0        # solute diffusivity [m²/s]
 *
 *   [output]
 *   format  = vtk       # vtk | csv | ppm | all
 *   dir     = output/grain_growth
 *   prefix  = snap
 *   compact = false     # CSV: write only interface cells
 *
 *   [grain_growth]      # simulation-specific section
 *   n_grains = 80
 *   seed     = 42
 *
 * Key rules:
 *   - Keys are looked up as "[section]/key" strings.
 *   - Values after '#' are treated as comments.
 *   - All values are stored as strings; typed getters parse on demand.
 *   - Missing keys trigger an exception with a helpful message.
 */

#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstdlib>

namespace gpk {

class InputParser {
public:
    explicit InputParser(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("InputParser: cannot open '" + path + "'");
        parse(f);
    }

    // ---- typed getters ----
    std::string get_str(const std::string& section, const std::string& key,
                        const std::string& def = "") const {
        auto it = data_.find(section + "/" + key);
        if (it == data_.end()) {
            if (def.empty()) throw missing(section, key);
            return def;
        }
        return it->second;
    }

    int get_int(const std::string& section, const std::string& key,
                int def = -999999) const {
        auto it = data_.find(section + "/" + key);
        if (it == data_.end()) {
            if (def == -999999) throw missing(section, key);
            return def;
        }
        return std::stoi(it->second);
    }

    double get_double(const std::string& section, const std::string& key,
                      double def = 1e300) const {
        auto it = data_.find(section + "/" + key);
        if (it == data_.end()) {
            if (def > 1e299) throw missing(section, key);
            return def;
        }
        return std::stod(it->second);
    }

    float get_float(const std::string& section, const std::string& key,
                    float def = 1e30f) const {
        return static_cast<float>(get_double(section, key,
                                             def < 1e29f ? (double)def : 1e300));
    }

    bool get_bool(const std::string& section, const std::string& key,
                  bool def = false) const {
        auto it = data_.find(section + "/" + key);
        if (it == data_.end()) return def;
        std::string v = it->second;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v == "true" || v == "1" || v == "yes";
    }

    bool has(const std::string& section, const std::string& key) const {
        return data_.count(section + "/" + key) > 0;
    }

private:
    std::unordered_map<std::string, std::string> data_;

    static std::string trim(const std::string& s) {
        auto b = s.find_first_not_of(" \t\r\n");
        if (b == std::string::npos) return "";
        auto e = s.find_last_not_of(" \t\r\n");
        return s.substr(b, e - b + 1);
    }

    void parse(std::ifstream& f) {
        std::string section, line;
        while (std::getline(f, line)) {
            // strip comment
            auto ci = line.find('#');
            if (ci != std::string::npos) line = line.substr(0, ci);
            line = trim(line);
            if (line.empty()) continue;

            if (line.front() == '[' && line.back() == ']') {
                section = trim(line.substr(1, line.size() - 2));
                continue;
            }

            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            data_[section + "/" + key] = val;
        }
    }

    static std::runtime_error missing(const std::string& sec,
                                       const std::string& key) {
        return std::runtime_error(
            "InputParser: missing required key [" + sec + "] " + key);
    }
};

// =========================================================================
// Unified output writer — dispatches to VTK / CSV / PPM / all
// =========================================================================

} // namespace gpk — continued below after including writers
