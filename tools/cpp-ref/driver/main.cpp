// Headless reference driver for the CCTag C++ library.
//
//   cctag_ref --image f.png [--nbrings 3] [--threads N] [--warmup W] [--iters K]
//             [--mode e2e|stages] [--dump-dir D] [--json out.json] [--csv out.csv]
//   cctag_ref --selftest-rng
//
// e2e:    times cctagDetection() end to end (steady_clock, microseconds).
// stages: times the public pipeline stages separately.
// dump:   writes the front-end planes / edge collections / markers in the
//         binary format read by cctag-rs (src/refdata.rs).
#include <cctag/Canny.hpp>
#include <cctag/CCTag.hpp>
#include <cctag/CCTagMarkersBank.hpp>
#include <cctag/Detection.hpp>
#include <cctag/EdgePoint.hpp>
#include <cctag/Identification.hpp>
#include <cctag/ImagePyramid.hpp>
#include <cctag/Level.hpp>
#include <cctag/Multiresolution.hpp>
#include <cctag/Params.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>
#include <cctag/utils/pcg_random.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/global_control.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static double us(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Stats
{
    double median = 0, p10 = 0, p90 = 0, min = 0, max = 0;
};

static Stats stats(std::vector<double> v)
{
    Stats s;
    if(v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    s.p10 = v[v.size() / 10];
    s.p90 = v[(v.size() * 9) / 10];
    s.min = v.front();
    s.max = v.back();
    return s;
}

// ---------------------------------------------------------------- dump format
static void writePlane(const std::string& path, const cv::Mat& m, uint32_t dtype)
{
    std::ofstream f(path, std::ios::binary);
    f.write("CCTD", 4);
    uint32_t w = m.cols, h = m.rows;
    f.write((const char*)&dtype, 4);
    f.write((const char*)&w, 4);
    f.write((const char*)&h, 4);
    for(int y = 0; y < m.rows; ++y)
        f.write((const char*)m.ptr(y), (size_t)m.cols * m.elemSize());
}

static void writeRecords(const std::string& path, const std::vector<int32_t>& flat, uint32_t nFields)
{
    std::ofstream f(path, std::ios::binary);
    uint32_t count = nFields ? (uint32_t)(flat.size() / nFields) : 0;
    f.write((const char*)&count, 4);
    f.write((const char*)flat.data(), flat.size() * 4);
}

static int32_t f2bits(float v)
{
    int32_t r;
    std::memcpy(&r, &v, 4);
    return r;
}

// Dump one edge point collection after vote (+ sorted seeds + loop-1 replica).
static void dumpCollection(const std::string& dir, int level, cctag::EdgePointCollection& coll,
                           const std::vector<cctag::EdgePoint*>& seeds, const cctag::Parameters& params, int rows)
{
    const int n = coll.get_point_count();
    std::vector<int32_t> pts, links, voff, vlist, sd, fl;
    pts.reserve((size_t)n * 4);
    links.reserve((size_t)n * 2);
    voff.reserve(n + 1);
    fl.reserve(n);
    for(int i = 0; i < n; ++i)
    {
        cctag::EdgePoint* p = coll(i);
        pts.push_back(p->x());
        pts.push_back(p->y());
        pts.push_back((int32_t)p->dX());
        pts.push_back((int32_t)p->dY());
        links.push_back(coll(coll.before(p)));
        links.push_back(coll(coll.after(p)));
        fl.push_back(f2bits(p->_flowLength));
    }
    int32_t off = 0;
    voff.push_back(0);
    for(int i = 0; i < n; ++i)
    {
        cctag::EdgePoint* p = coll(i);
        auto v = coll.voters(p);
        for(auto it = v.first; it != v.second; ++it) vlist.push_back(*it);
        off += (int32_t)(v.second - v.first);
        voff.push_back(off);
    }
    for(cctag::EdgePoint* s : seeds)
    {
        sd.push_back(coll(s));
        sd.push_back(s->_isMax);
    }
    const std::string pre = dir + "/L" + std::to_string(level) + "_";
    writeRecords(pre + "points.bin", pts, 4);
    writeRecords(pre + "links.bin", links, 2);
    writeRecords(pre + "voters_off.bin", voff, 1);
    writeRecords(pre + "voters.bin", vlist, 1);
    writeRecords(pre + "seeds.bin", sd, 2);
    writeRecords(pre + "flow_length.bin", fl, 1);

    // Loop-1 replica (constructFlowComponentFromSeed, serial): segments + children.
    // NOTE: mutates processed_in; the caller must not run detection on this collection afterwards.
    const std::size_t nMax = std::max(rows / 2, (int)params._maximumNbSeeds);
    const std::size_t nSeeds = std::min(seeds.size(), nMax);
    std::vector<int32_t> segs;   // records: seed idx, avgVote bits, nSegment, nChildren, then indices...
    std::vector<int32_t> flat;   // per candidate: [seed, avgbits, segLen, childLen, seg..., children...]
    for(std::size_t i = 0; i < nSeeds; ++i)
    {
        cctag::EdgePoint* seed = seeds[i];
        if(coll.test_processed_in(seed)) continue;
        std::list<cctag::EdgePoint*> seg;
        cctag::edgeLinking(coll, seg, seed, params._windowSizeOnInnerEllipticSegment, params._averageVoteMin);
        int nReceived = 0, nVoted = 0;
        for(cctag::EdgePoint* p : seg)
        {
            int vs = coll.voters_size(p);
            nReceived += vs;
            if(vs > 0) ++nVoted;
        }
        float avg = (float)(nReceived * nReceived) / (float)nVoted;
        std::list<cctag::EdgePoint*> children;
        cctag::childrenOf(coll, seg, children);
        flat.push_back(coll(seed));
        flat.push_back(f2bits(avg));
        flat.push_back((int32_t)seg.size());
        flat.push_back((int32_t)children.size());
        for(cctag::EdgePoint* p : seg) flat.push_back(coll(p));
        for(cctag::EdgePoint* p : children) flat.push_back(coll(p));
    }
    writeRecords(pre + "loop1.bin", flat, 1);
}

static void jsonFloat(std::ostream& o, float v)
{
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.9g", (double)v);
    o << buf;
}

static void jsonEllipse(std::ostream& o, const cctag::numerical::geometry::Ellipse& e)
{
    o << "{\"cx\":";
    jsonFloat(o, e.center().x());
    o << ",\"cy\":";
    jsonFloat(o, e.center().y());
    o << ",\"a\":";
    jsonFloat(o, e.a());
    o << ",\"b\":";
    jsonFloat(o, e.b());
    o << ",\"angle\":";
    jsonFloat(o, e.angle());
    o << "}";
}

static void jsonMarkers(std::ostream& o, const cctag::CCTag::List& markers, bool final)
{
    o << "[";
    bool first = true;
    for(const cctag::CCTag& m : markers)
    {
        if(!first) o << ",";
        first = false;
        o << "{\"id\":" << m.id() << ",\"status\":" << m.getStatus() << ",\"x\":";
        jsonFloat(o, m.x());
        o << ",\"y\":";
        jsonFloat(o, m.y());
        o << ",\"quality\":";
        jsonFloat(o, m.quality());
        o << ",\"level\":" << m.pyramidLevel() << ",\"scale\":";
        jsonFloat(o, m.scale());
        o << ",\"outer\":";
        jsonEllipse(o, m.outerEllipse());
        o << ",\"rescaled\":";
        jsonEllipse(o, m.rescaledOuterEllipse());
        o << ",\"n_rescaled_points\":" << m.rescaledOuterEllipsePoints().size();
        o << ",\"n_points\":[";
        for(std::size_t i = 0; i < m.points().size(); ++i)
        {
            if(i) o << ",";
            o << m.points()[i].size();
        }
        o << "]";
        if(final)
        {
            o << ",\"H\":[";
            for(int r = 0; r < 3; ++r)
                for(int c = 0; c < 3; ++c)
                {
                    if(r || c) o << ",";
                    jsonFloat(o, m.homography()(r, c));
                }
            o << "]";
        }
        o << "}";
    }
    o << "]";
}

int main(int argc, char** argv)
{
    std::string image, dumpDir, jsonPath, csvPath, mode = "e2e", bankFile;
    int nRings = 3, threads = 0, warmup = 3, iters = 20;
    bool selftestRng = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if(a == "--image") image = next();
        else if(a == "--nbrings" || a == "-n") nRings = std::stoi(next());
        else if(a == "--threads") threads = std::stoi(next());
        else if(a == "--warmup") warmup = std::stoi(next());
        else if(a == "--iters") iters = std::stoi(next());
        else if(a == "--mode") mode = next();
        else if(a == "--dump-dir") dumpDir = next();
        else if(a == "--json") jsonPath = next();
        else if(a == "--csv") csvPath = next();
        else if(a == "--bank") bankFile = next();
        else if(a == "--selftest-rng") selftestRng = true;
        else
        {
            std::cerr << "unknown arg " << a << std::endl;
            return 2;
        }
    }

    if(selftestRng)
    {
        pcg32 rng(271828);
        std::cout << "raw";
        for(int i = 0; i < 64; ++i) std::cout << " " << rng();
        std::cout << "\n";
        for(uint32_t N : {7u, 60u, 150u, 1000u, 24963u})
        {
            pcg32 r2(271828);
            std::cout << "bounded " << N;
            for(int i = 0; i < 16; ++i) std::cout << " " << r2(N);
            std::cout << "\n";
        }
        std::array<int, 5> perm{};
        std::cout << "rand_5_k 60";
        for(int k = 0; k < 4; ++k)
        {
            cctag::numerical::rand_5_k(perm, 60);
            for(int v : perm) std::cout << " " << v;
        }
        std::cout << "\n";
        return 0;
    }
    if(image.empty())
    {
        std::cerr << "usage: cctag_ref --image f.png [--threads N] [--warmup W] [--iters K] [--mode e2e|stages] [--dump-dir D] [--json out]" << std::endl;
        return 2;
    }

    std::unique_ptr<tbb::global_control> gc;
    if(threads > 0)
    {
        gc.reset(new tbb::global_control(tbb::global_control::max_allowed_parallelism, threads));
        cv::setNumThreads(threads);
    }

    cv::Mat color = cv::imread(image, cv::IMREAD_COLOR);
    if(color.empty())
    {
        std::cerr << "cannot read " << image << std::endl;
        return 1;
    }
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    cctag::Parameters params(nRings);
    std::unique_ptr<cctag::CCTagMarkersBank> bank(bankFile.empty() ? new cctag::CCTagMarkersBank(nRings)
                                                                   : new cctag::CCTagMarkersBank(bankFile));

    std::map<std::string, std::vector<double>> timings;
    cctag::CCTag::List lastMarkers;

    if(!dumpDir.empty())
    {
        writePlane(dumpDir + "/gray.bin", gray, 1);
        // Pass A: front-end + vote + loop-1 replica on private collections.
        cctag::ImagePyramid pyr(gray.cols, gray.rows, params._numberOfProcessedMultiresLayers, false);
        pyr.build(gray, params._cannyThrLow, params._cannyThrHigh, &params);
        for(int i = 0; i < (int)params._numberOfProcessedMultiresLayers; ++i)
        {
            cctag::Level* lvl = pyr.getLevel(i);
            const std::string pre = dumpDir + "/L" + std::to_string(i) + "_";
            writePlane(pre + "src.bin", lvl->getSrc(), 1);
            writePlane(pre + "dx.bin", lvl->getDx(), 2);
            writePlane(pre + "dy.bin", lvl->getDy(), 2);
            writePlane(pre + "edges.bin", lvl->getEdges(), 1);
            cctag::EdgePointCollection coll(gray.cols, gray.rows);
            cctag::edgesPointsFromCanny(coll, lvl->getEdges(), lvl->getDx(), lvl->getDy());
            std::vector<cctag::EdgePoint*> seeds;
            cctag::vote(coll, seeds, lvl->getDx(), lvl->getDy(), params);
            if(seeds.size() > 1) std::stable_sort(seeds.begin(), seeds.end(), cctag::receivedMoreVoteThan);
            dumpCollection(dumpDir, i, coll, seeds, params, lvl->getSrc().rows);
        }
        // Pass B: full pipeline with the pieces exposed, dumping pre- and post-identification markers.
#ifdef CCTAG_PARITY_DIRECT_FILTER
        cctag::numerical::resetRng();
#endif
        std::srand(1);
        cctag::ImagePyramid pyr2(gray.cols, gray.rows, params._numberOfProcessedMultiresLayers, false);
        pyr2.build(gray, params._cannyThrLow, params._cannyThrHigh, &params);
        cctag::CCTag::List markers;
        cctag::cctagMultiresDetection(markers, gray, pyr2, 0, nullptr, params, nullptr);
        std::ofstream jm(dumpDir + "/markers_pre_ident.json");
        jsonMarkers(jm, markers, false);
        jm << "\n";
#ifdef CCTAG_PARITY_DIRECT_FILTER
        {
            std::ofstream jr(dumpDir + "/rng_draws.txt");
            jr << cctag::numerical::rngDrawCount() << "\n";
        }
#endif
        // identification (as Detection.cpp:872-964)
        {
            const std::size_t numTags = markers.size();
            std::vector<std::vector<cctag::ImageCut>> cuts(numTags);
            std::vector<int> detected(numTags, -1);
            int t = 0;
            for(const cctag::CCTag& c : markers)
            {
                detected[t] = cctag::identification::identify_step_1(t, c, cuts[t], pyr2.getLevel(0)->getSrc(), params);
                ++t;
            }
            t = 0;
            for(cctag::CCTag& c : markers)
            {
                if(detected[t] == cctag::status::id_reliable)
                    detected[t] = cctag::identification::identify_step_2(t, c, cuts[t], bank->getMarkers(),
                                                                         pyr2.getLevel(0)->getSrc(), nullptr, params);
                c.setStatus(detected[t]);
                ++t;
            }
        }
        std::ofstream jf(dumpDir + "/markers_post_ident.json");
        jsonMarkers(jf, markers, true);
        jf << "\n";
        // dedup as upstream
        cctag::CCTag::List prelim, fin;
        for(const cctag::CCTag& m : markers) cctag::update(prelim, m);
        for(const cctag::CCTag& m : prelim) cctag::update(fin, m);
        fin.sort();
        std::ofstream jd(dumpDir + "/markers_final.json");
        jsonMarkers(jd, fin, true);
        jd << "\n";
        lastMarkers = fin;
    }

    if(mode == "e2e")
    {
        for(int it = 0; it < warmup + iters; ++it)
        {
            cctag::CCTag::List markers;
            auto t0 = Clock::now();
            cctag::cctagDetection(markers, 0, it, gray, params, *bank, true, nullptr);
            auto t1 = Clock::now();
            if(it >= warmup) timings["e2e"].push_back(us(t0, t1));
            if(it == warmup + iters - 1) lastMarkers = markers;
        }
    }
    else if(mode == "stages")
    {
        for(int it = 0; it < warmup + iters; ++it)
        {
            const bool rec = it >= warmup;
            auto push = [&](const std::string& k, double v) { if(rec) timings[k].push_back(v); };
            auto t0 = Clock::now();
            cctag::ImagePyramid pyr(gray.cols, gray.rows, params._numberOfProcessedMultiresLayers, false);
            pyr.build(gray, params._cannyThrLow, params._cannyThrHigh, &params);
            auto t1 = Clock::now();
            push("pyramid", us(t0, t1));
            // per-level stages on private collections (same work as cctagMultiresDetection_inner)
            for(int i = (int)params._numberOfProcessedMultiresLayers - 1; i >= 0; --i)
            {
                cctag::Level* lvl = pyr.getLevel(i);
                const std::string L = "L" + std::to_string(i) + "/";
                auto a0 = Clock::now();
                cctag::EdgePointCollection coll(gray.cols, gray.rows);
                auto a1 = Clock::now();
                cctag::edgesPointsFromCanny(coll, lvl->getEdges(), lvl->getDx(), lvl->getDy());
                auto a2 = Clock::now();
                std::vector<cctag::EdgePoint*> seeds;
                cctag::vote(coll, seeds, lvl->getDx(), lvl->getDy(), params);
                if(seeds.size() > 1) std::sort(seeds.begin(), seeds.end(), cctag::receivedMoreVoteThan);
                auto a3 = Clock::now();
                cctag::CCTag::List lm;
                cctag::cctagDetectionFromEdges(lm, coll, lvl->getSrc(), seeds, 0, i, std::pow(2.0, i), params, nullptr);
                auto a4 = Clock::now();
                push(L + "alloc", us(a0, a1));
                push(L + "edges", us(a1, a2));
                push(L + "vote", us(a2, a3));
                push(L + "from_edges", us(a3, a4));
            }
            auto t2 = Clock::now();
            cctag::CCTag::List markers;
            cctag::cctagMultiresDetection(markers, gray, pyr, 0, nullptr, params, nullptr);
            auto t3 = Clock::now();
            push("multires", us(t2, t3));
            {
                const std::size_t numTags = markers.size();
                std::vector<std::vector<cctag::ImageCut>> cuts(numTags);
                std::vector<int> detected(numTags, -1);
                int t = 0;
                for(const cctag::CCTag& c : markers)
                {
                    detected[t] = cctag::identification::identify_step_1(t, c, cuts[t], pyr.getLevel(0)->getSrc(), params);
                    ++t;
                }
                t = 0;
                for(cctag::CCTag& c : markers)
                {
                    if(detected[t] == cctag::status::id_reliable)
                        detected[t] = cctag::identification::identify_step_2(t, c, cuts[t], bank->getMarkers(),
                                                                             pyr.getLevel(0)->getSrc(), nullptr, params);
                    c.setStatus(detected[t]);
                    ++t;
                }
            }
            auto t4 = Clock::now();
            push("identification", us(t3, t4));
            if(it == warmup + iters - 1) lastMarkers = markers;
        }
    }

    // report
    std::ostream* csv = nullptr;
    std::ofstream csvf;
    if(!csvPath.empty())
    {
        csvf.open(csvPath);
        csv = &csvf;
        *csv << "stage,median_us,p10_us,p90_us,min_us,max_us,n\n";
    }
    for(auto& kv : timings)
    {
        Stats s = stats(kv.second);
        std::printf("%-20s median %10.1f us  p10 %10.1f  p90 %10.1f  min %10.1f  max %10.1f  (n=%zu)\n", kv.first.c_str(),
                    s.median, s.p10, s.p90, s.min, s.max, kv.second.size());
        if(csv)
            *csv << kv.first << "," << s.median << "," << s.p10 << "," << s.p90 << "," << s.min << "," << s.max << ","
                 << kv.second.size() << "\n";
    }
    std::cout << "markers " << lastMarkers.size() << "\n";
    for(const cctag::CCTag& m : lastMarkers)
        std::printf("%.9g %.9g %d %d %.9g %d\n", (double)m.x(), (double)m.y(), m.id(), m.getStatus(), (double)m.quality(),
                    m.pyramidLevel());
    if(!jsonPath.empty())
    {
        std::ofstream j(jsonPath);
        jsonMarkers(j, lastMarkers, true);
        j << "\n";
    }
    return 0;
}
