#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kMadToSigma = 1.4826;

constexpr int kWindowSize = 50;
constexpr int kWindowStride = 1;
constexpr int kMinStableSamples = 100;

constexpr double kTranslationDispersionLimit = 0.01;
constexpr double kRotationDispersionLimitRad = 0.5 * kDegToRad;
constexpr double kWindowTranslationChangeLimit = 0.003;
constexpr double kWindowRotationChangeLimitRad = 0.15 * kDegToRad;
constexpr double kTranslationGateFloor = 0.005;
constexpr double kPoseTranslationGateFloor = 0.005;
constexpr double kPoseRotationGateFloor = 0.15 * kDegToRad;
constexpr size_t kMinFinalInliers = 30;

struct Sample {
    double frame_id = 0.0;
    double fitness = 0.0;
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct WindowStats {
    size_t begin = 0;
    size_t end = 0;
    Eigen::Vector3d center_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond center_q = Eigen::Quaterniond::Identity();
    double translation_dispersion = 0.0;
    double rotation_dispersion = 0.0;
    bool dispersion_ok = false;
};

struct StableInterval {
    size_t begin = 0;
    size_t end = 0;
    bool found = false;
};

double Clamp(double value, double low, double high)
{
    return std::max(low, std::min(value, high));
}

double Median(std::vector<double> values)
{
    if (values.empty()) {
        return 0.0;
    }

    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];

    if (values.size() % 2 == 1) {
        return upper;
    }

    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return 0.5 * (values[mid - 1] + upper);
}

double Mad(const std::vector<double>& values, double center)
{
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) {
        deviations.push_back(std::abs(value - center));
    }
    return Median(deviations);
}

Eigen::Quaterniond QuaternionFromRpy(double roll, double pitch, double yaw)
{
    Eigen::Quaterniond q =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    return q.normalized();
}

Eigen::Vector3d RpyFromQuaternion(const Eigen::Quaterniond& q)
{
    const Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    const double sy = Clamp(-R(2, 0), -1.0, 1.0);
    const double pitch = std::asin(sy);
    const double cos_pitch = std::cos(pitch);

    double roll = 0.0;
    double yaw = 0.0;
    if (std::abs(cos_pitch) > 1e-12) {
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw = std::atan2(R(1, 0), R(0, 0));
    } else {
        roll = 0.0;
        yaw = std::atan2(-R(0, 1), R(1, 1));
    }

    return Eigen::Vector3d(roll, pitch, yaw);
}

double QuaternionAngle(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b)
{
    const double dot = Clamp(std::abs(a.normalized().dot(b.normalized())), 0.0, 1.0);
    return 2.0 * std::acos(dot);
}

Eigen::Quaterniond MarkleyAverage(const std::vector<const Sample*>& samples)
{
    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    const Eigen::Quaterniond q_ref = samples.front()->q.normalized();

    for (const Sample* sample : samples) {
        Eigen::Quaterniond q = sample->q.normalized();
        if (q.dot(q_ref) < 0.0) {
            q.coeffs() *= -1.0;
        }

        const Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
        A += v * v.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(A);
    const Eigen::Vector4d v = solver.eigenvectors().col(3);
    Eigen::Quaterniond q_mean(v(0), v(1), v(2), v(3));
    q_mean.normalize();

    if (q_mean.dot(q_ref) < 0.0) {
        q_mean.coeffs() *= -1.0;
    }
    return q_mean;
}

Eigen::Vector3d TranslationMean(const std::vector<const Sample*>& samples)
{
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const Sample* sample : samples) {
        mean += sample->t;
    }
    return mean / static_cast<double>(samples.size());
}

Eigen::Vector3d TranslationMedian(const std::vector<const Sample*>& samples)
{
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> zs;
    xs.reserve(samples.size());
    ys.reserve(samples.size());
    zs.reserve(samples.size());

    for (const Sample* sample : samples) {
        xs.push_back(sample->t.x());
        ys.push_back(sample->t.y());
        zs.push_back(sample->t.z());
    }

    return Eigen::Vector3d(Median(xs), Median(ys), Median(zs));
}

std::vector<const Sample*> MakeSamplePtrs(
    const std::vector<Sample>& samples,
    size_t begin,
    size_t end)
{
    std::vector<const Sample*> result;
    result.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        result.push_back(&samples[i]);
    }
    return result;
}

WindowStats ComputeWindowStats(const std::vector<Sample>& samples, size_t begin)
{
    WindowStats stats;
    stats.begin = begin;
    stats.end = begin + kWindowSize;

    const std::vector<const Sample*> window_samples =
        MakeSamplePtrs(samples, stats.begin, stats.end);

    stats.center_t = TranslationMedian(window_samples);
    stats.center_q = MarkleyAverage(window_samples);

    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    translation_errors.reserve(window_samples.size());
    rotation_errors.reserve(window_samples.size());

    for (const Sample* sample : window_samples) {
        translation_errors.push_back((sample->t - stats.center_t).norm());
        rotation_errors.push_back(QuaternionAngle(stats.center_q, sample->q));
    }

    stats.translation_dispersion = Median(translation_errors);
    stats.rotation_dispersion = Median(rotation_errors);
    stats.dispersion_ok =
        stats.translation_dispersion < kTranslationDispersionLimit &&
        stats.rotation_dispersion < kRotationDispersionLimitRad;

    return stats;
}

void FinalizeStableRun(
    const std::vector<WindowStats>& windows,
    size_t run_begin,
    size_t run_end,
    StableInterval* best)
{
    if (run_begin > run_end) {
        return;
    }

    StableInterval candidate;
    candidate.begin = windows[run_begin].begin;
    candidate.end = windows[run_end].end;
    candidate.found = candidate.end - candidate.begin >= kMinStableSamples;

    if (!candidate.found) {
        return;
    }

    const size_t candidate_count = candidate.end - candidate.begin;
    const size_t best_count = best->found ? best->end - best->begin : 0;
    if (!best->found ||
        candidate_count > best_count ||
        (candidate_count == best_count && candidate.begin > best->begin)) {
        *best = candidate;
    }
}

StableInterval FindStableInterval(const std::vector<Sample>& samples)
{
    StableInterval best;
    if (samples.size() < static_cast<size_t>(kMinStableSamples)) {
        return best;
    }

    const size_t analysis_begin = samples.size() / 2;
    if (analysis_begin + kWindowSize > samples.size()) {
        return best;
    }

    std::vector<WindowStats> windows;
    for (size_t begin = analysis_begin; begin + kWindowSize <= samples.size();
         begin += kWindowStride) {
        windows.push_back(ComputeWindowStats(samples, begin));
    }

    bool in_run = false;
    size_t run_begin = 0;
    size_t run_end = 0;

    for (size_t i = 0; i < windows.size(); ++i) {
        if (!windows[i].dispersion_ok) {
            if (in_run) {
                FinalizeStableRun(windows, run_begin, run_end, &best);
                in_run = false;
            }
            continue;
        }

        if (!in_run) {
            in_run = true;
            run_begin = i;
            run_end = i;
            continue;
        }

        const double translation_change =
            (windows[i].center_t - windows[run_end].center_t).norm();
        const double rotation_change =
            QuaternionAngle(windows[i].center_q, windows[run_end].center_q);

        if (translation_change < kWindowTranslationChangeLimit &&
            rotation_change < kWindowRotationChangeLimitRad) {
            run_end = i;
        } else {
            FinalizeStableRun(windows, run_begin, run_end, &best);
            run_begin = i;
            run_end = i;
        }
    }

    if (in_run) {
        FinalizeStableRun(windows, run_begin, run_end, &best);
    }

    return best;
}

std::vector<const Sample*> FilterByFitness(const std::vector<const Sample*>& samples)
{
    std::vector<double> fitness_values;
    fitness_values.reserve(samples.size());
    for (const Sample* sample : samples) {
        fitness_values.push_back(sample->fitness);
    }

    const double fitness_median = Median(fitness_values);
    const double fitness_mad = Mad(fitness_values, fitness_median);
    const double fitness_threshold = fitness_median + 3.0 * kMadToSigma * fitness_mad;

    std::vector<const Sample*> inliers;
    inliers.reserve(samples.size());
    for (const Sample* sample : samples) {
        if (sample->fitness <= fitness_threshold) {
            inliers.push_back(sample);
        }
    }

    return inliers;
}

std::vector<const Sample*> FilterByTranslationMad(
    const std::vector<const Sample*>& samples,
    Eigen::Vector3d* center,
    Eigen::Vector3d* mad)
{
    *center = TranslationMedian(samples);

    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<double> dz;
    dx.reserve(samples.size());
    dy.reserve(samples.size());
    dz.reserve(samples.size());

    for (const Sample* sample : samples) {
        dx.push_back(std::abs(sample->t.x() - center->x()));
        dy.push_back(std::abs(sample->t.y() - center->y()));
        dz.push_back(std::abs(sample->t.z() - center->z()));
    }

    *mad = Eigen::Vector3d(Median(dx), Median(dy), Median(dz));
    const Eigen::Vector3d threshold =
        3.0 * kMadToSigma *
        mad->array().max(kTranslationGateFloor / (3.0 * kMadToSigma)).matrix();

    std::vector<const Sample*> inliers;
    inliers.reserve(samples.size());
    for (const Sample* sample : samples) {
        const Eigen::Vector3d error = (sample->t - *center).cwiseAbs();
        if (error.x() <= threshold.x() &&
            error.y() <= threshold.y() &&
            error.z() <= threshold.z()) {
            inliers.push_back(sample);
        }
    }

    return inliers;
}

std::vector<const Sample*> FilterByFullPose(
    const std::vector<const Sample*>& samples,
    const Eigen::Vector3d& t_mean,
    const Eigen::Quaterniond& q_mean)
{
    std::vector<double> translation_errors;
    std::vector<double> rotation_errors;
    translation_errors.reserve(samples.size());
    rotation_errors.reserve(samples.size());

    for (const Sample* sample : samples) {
        translation_errors.push_back((sample->t - t_mean).norm());
        rotation_errors.push_back(QuaternionAngle(q_mean, sample->q));
    }

    const double t_error_median = Median(translation_errors);
    const double r_error_median = Median(rotation_errors);
    const double t_error_mad = Mad(translation_errors, t_error_median);
    const double r_error_mad = Mad(rotation_errors, r_error_median);
    const double t_threshold = std::max(
        t_error_median + 3.0 * kMadToSigma * t_error_mad,
        kPoseTranslationGateFloor);
    const double r_threshold = std::max(
        r_error_median + 3.0 * kMadToSigma * r_error_mad,
        kPoseRotationGateFloor);

    std::vector<const Sample*> inliers;
    inliers.reserve(samples.size());
    for (const Sample* sample : samples) {
        const double t_error = (sample->t - t_mean).norm();
        const double r_error = QuaternionAngle(q_mean, sample->q);
        if (t_error <= t_threshold && r_error <= r_threshold) {
            inliers.push_back(sample);
        }
    }

    return inliers;
}

double Stddev(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }

    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());

    double variance = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        variance += diff * diff;
    }

    return std::sqrt(variance / static_cast<double>(values.size()));
}

bool ReadSamples(const std::string& path, std::vector<Sample>* samples)
{
    std::ifstream calib_file(path.c_str());
    if (!calib_file.is_open()) {
        std::cerr << "ERROR: cannot open " << path << std::endl;
        return false;
    }

    double frame_id = 0.0;
    double fitness = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    while (calib_file >> frame_id >> fitness >> x >> y >> z >> roll >> pitch >> yaw) {
        Sample sample;
        sample.frame_id = frame_id;
        sample.fitness = fitness;
        sample.t = Eigen::Vector3d(x, y, z);
        sample.q = QuaternionFromRpy(roll, pitch, yaw);
        samples->push_back(sample);
    }

    return true;
}

bool ReadInitialGuess(const std::string& path, Eigen::Vector3d* t, Eigen::Quaterniond* q)
{
    std::ifstream init_file(path.c_str());
    if (!init_file.is_open()) {
        std::cerr << "WARNING: cannot open " << path << std::endl;
        return false;
    }

    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            if (!(init_file >> matrix(row, col))) {
                std::cerr << "WARNING: cannot read a 4x4 matrix from " << path << std::endl;
                return false;
            }
        }
    }

    *t = matrix.block<3, 1>(0, 3);
    *q = Eigen::Quaterniond(matrix.block<3, 3>(0, 0)).normalized();
    return true;
}

}  // namespace

int main()
{
    const std::string calib_path = "../data/calib_data.txt";

    std::vector<Sample> samples;
    if (!ReadSamples(calib_path, &samples)) {
        return 1;
    }

    Eigen::Vector3d init_t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond init_q = Eigen::Quaterniond::Identity();
    const bool have_init = ReadInitialGuess("../data/Init_Matrix.txt", &init_t, &init_q);

    std::cout << "Read " << samples.size() << " data" << std::endl;

    const StableInterval stable = FindStableInterval(samples);
    const std::vector<const Sample*> selected_samples =
        MakeSamplePtrs(samples, 0, samples.size());
    const std::vector<const Sample*> fitness_inliers =
        FilterByFitness(selected_samples);

    if (fitness_inliers.empty()) {
        std::cerr << "ERROR: all samples rejected by fitness gate" << std::endl;
        return 1;
    }

    Eigen::Vector3d translation_gate_center = Eigen::Vector3d::Zero();
    Eigen::Vector3d translation_gate_mad = Eigen::Vector3d::Zero();
    const std::vector<const Sample*> translation_inliers =
        FilterByTranslationMad(
            fitness_inliers, &translation_gate_center, &translation_gate_mad);

    if (translation_inliers.empty()) {
        std::cerr << "ERROR: all samples rejected by translation robust gate" << std::endl;
        return 1;
    }

    const Eigen::Vector3d first_t_mean = TranslationMean(translation_inliers);
    const Eigen::Quaterniond first_q_mean = MarkleyAverage(translation_inliers);
    const std::vector<const Sample*> final_inliers =
        FilterByFullPose(translation_inliers, first_t_mean, first_q_mean);

    if (final_inliers.empty()) {
        std::cerr << "ERROR: all samples rejected by full-pose robust gate" << std::endl;
        return 1;
    }

    bool using_initial_guess_fallback = false;
    Eigen::Vector3d final_t = TranslationMedian(final_inliers);
    Eigen::Quaterniond final_q = MarkleyAverage(final_inliers);
    if (have_init && final_inliers.size() < kMinFinalInliers) {
        using_initial_guess_fallback = true;
        final_t = init_t;
        final_q = init_q;
    }
    const Eigen::Vector3d final_rpy = RpyFromQuaternion(final_q);

    std::vector<double> final_translation_errors;
    std::vector<double> final_rotation_errors;
    final_translation_errors.reserve(final_inliers.size());
    final_rotation_errors.reserve(final_inliers.size());

    for (const Sample* sample : final_inliers) {
        final_translation_errors.push_back((sample->t - final_t).norm());
        final_rotation_errors.push_back(QuaternionAngle(final_q, sample->q));
    }

    const double translation_error_median = Median(final_translation_errors);
    const double rotation_error_median = Median(final_rotation_errors);
    const double translation_mad =
        Mad(final_translation_errors, translation_error_median);
    const double rotation_mad =
        Mad(final_rotation_errors, rotation_error_median);

    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
    tf.block<3, 3>(0, 0) = final_q.normalized().toRotationMatrix();
    tf.block<3, 1>(0, 3) = final_t;

    const int fitness_rejected =
        static_cast<int>(selected_samples.size() - fitness_inliers.size());
    const int pose_rejected =
        static_cast<int>(fitness_inliers.size() - final_inliers.size());
    const int translation_gate_rejected =
        static_cast<int>(fitness_inliers.size() - translation_inliers.size());
    const int full_pose_gate_rejected =
        static_cast<int>(translation_inliers.size() - final_inliers.size());

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "----------------------------------" << std::endl;
    std::cout << "candidate selection:" << std::endl;
    std::cout << "  mode: all independent frame estimates" << std::endl;
    std::cout << "  sample count: " << selected_samples.size() << std::endl;
    if (stable.found) {
        std::cout << "  diagnostic stable interval: "
                  << samples[stable.begin].frame_id << " / "
                  << samples[stable.end - 1].frame_id << std::endl;
    } else {
        std::cout << "  diagnostic stable interval: not required / not found" << std::endl;
    }
    std::cout << "  fitness rejected count: " << fitness_rejected << std::endl;
    std::cout << "  pose rejected count: " << pose_rejected
              << " (translation gate " << translation_gate_rejected
              << ", full-pose gate " << full_pose_gate_rejected << ")"
              << std::endl;
    std::cout << "  final inlier count: " << final_inliers.size() << std::endl;
    if (using_initial_guess_fallback) {
        std::cout << "  confidence: LOW, fallback to Init_Matrix.txt" << std::endl;
    }
    std::cout << "  translation stddev / MAD (m): "
              << Stddev(final_translation_errors) << " / "
              << translation_mad << std::endl;
    std::cout << "  rotation angular stddev / MAD (deg): "
              << Stddev(final_rotation_errors) * kRadToDeg << " / "
              << rotation_mad * kRadToDeg << std::endl;

    std::cout << "----------------------------------" << std::endl;
    std::cout << "Final Result:" << std::endl;
    std::cout << "  x y z roll pitch yaw (rad): "
              << final_t.x() << " "
              << final_t.y() << " "
              << final_t.z() << " "
              << final_rpy.x() << " "
              << final_rpy.y() << " "
              << final_rpy.z() << std::endl;
    std::cout << "  RPY degree: "
              << final_rpy.x() * kRadToDeg << " "
              << final_rpy.y() * kRadToDeg << " "
              << final_rpy.z() * kRadToDeg << std::endl;

    std::cout << "----------------------------------" << std::endl;
    std::cout << "4x4 Result Matrix:" << std::endl;
    std::cout << tf << std::endl;

    std::cout << "----------------------------------" << std::endl;
    std::cout << "Adaptive-LIO:" << std::endl;
    std::cout << "extrinsic_xyz: ["
              << final_t.x() << ", "
              << final_t.y() << ", "
              << final_t.z() << "]" << std::endl;
    std::cout << "extrinsic_rpy_deg: ["
              << final_rpy.x() * kRadToDeg << ", "
              << final_rpy.y() * kRadToDeg << ", "
              << final_rpy.z() * kRadToDeg << "]" << std::endl;

    return 0;
}
