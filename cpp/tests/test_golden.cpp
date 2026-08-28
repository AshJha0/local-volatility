// Golden-value suite: loads ../data/golden/golden.json and asserts every
// case within its stated absolute tolerance, dispatching on name prefixes
// exactly as API_SPEC.md section 9 prescribes.

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "localvol/dupire.hpp"
#include "localvol/market.hpp"
#include "localvol/mc.hpp"
#include "localvol/pde.hpp"
#include "localvol/surface.hpp"
#include "mini_json.hpp"

using localvol::DupireLocalVol;
using localvol::ImpliedVolSurface;
using localvol::Market;
using localvol::McResult;
using localvol::McSettings;
using localvol::PdeSettings;
using localvol::price_american_put_pde;
using localvol::price_european_mc;
using localvol::price_european_pde;
using localvol::price_up_out_call_mc;

namespace {

const std::string kDataDir = LOCALVOL_DATA_DIR;

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

class GoldenTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        std::ifstream in(kDataDir + "/golden/golden.json");
        ASSERT_TRUE(in) << "cannot open golden.json";
        std::ostringstream ss;
        ss << in.rdbuf();
        root_ = mini_json::parse(ss.str());
        bundled_ = std::make_unique<ImpliedVolSurface>(
            ImpliedVolSurface::from_csv(kDataDir + "/implied_surface.csv"));
        flat_ = std::make_unique<ImpliedVolSurface>(
            ImpliedVolSurface::from_csv(kDataDir + "/flat_surface.csv"));
    }
    static void TearDownTestSuite() {
        bundled_.reset();
        flat_.reset();
        root_.reset();
    }

    static mini_json::ValuePtr root_;
    static std::unique_ptr<ImpliedVolSurface> bundled_;
    static std::unique_ptr<ImpliedVolSurface> flat_;
};

mini_json::ValuePtr GoldenTest::root_;
std::unique_ptr<ImpliedVolSurface> GoldenTest::bundled_;
std::unique_ptr<ImpliedVolSurface> GoldenTest::flat_;

double run_case(const mini_json::Value& c, const ImpliedVolSurface& bundled,
                const ImpliedVolSurface& flat) {
    const std::string name = c.at("name").as_string();
    const mini_json::Value& in = c.at("inputs");

    if (starts_with(name, "flat_dupire_")) {
        const DupireLocalVol lv(flat);
        return lv.vol(in.at("k").as_number(), in.at("T").as_number());
    }
    if (starts_with(name, "dupire_bundled_")) {
        const DupireLocalVol lv(bundled);
        return lv.vol(in.at("k").as_number(), in.at("T").as_number());
    }
    if (starts_with(name, "pde_flat_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        PdeSettings s;
        s.num_space = static_cast<int>(in.at("num_space").as_number());
        s.num_time = static_cast<int>(in.at("num_time").as_number());
        return price_european_pde(mkt, in.at("k").as_number(), in.at("T").as_number(),
                                  in.at("sigma").as_number(),
                                  in.at("call").as_number() != 0.0, s);
    }
    if (starts_with(name, "pde_localvol_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        const DupireLocalVol lv(bundled);
        PdeSettings s;
        s.num_space = static_cast<int>(in.at("num_space").as_number());
        s.num_time = static_cast<int>(in.at("num_time").as_number());
        return price_european_pde(
            mkt, in.at("k").as_number(), in.at("T").as_number(),
            [&lv](double k, double t) { return lv.vol(k, t); }, true, s,
            in.at("sigma_ref").as_number());
    }
    if (starts_with(name, "american_put_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        PdeSettings s;
        s.num_space = static_cast<int>(in.at("num_space").as_number());
        s.num_time = static_cast<int>(in.at("num_time").as_number());
        return price_american_put_pde(mkt, in.at("k").as_number(), in.at("T").as_number(),
                                      in.at("sigma").as_number(), s);
    }
    if (starts_with(name, "mc_flat_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        McSettings s;
        s.n_paths = static_cast<int>(in.at("n_paths").as_number());
        s.n_steps = static_cast<int>(in.at("n_steps").as_number());
        return price_european_mc(mkt, in.at("k").as_number(), in.at("T").as_number(),
                                 in.at("sigma").as_number(), true, s)
            .price;
    }
    if (starts_with(name, "mc_localvol_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        const DupireLocalVol lv(bundled);
        McSettings s;
        s.n_paths = static_cast<int>(in.at("n_paths").as_number());
        s.n_steps = static_cast<int>(in.at("n_steps").as_number());
        return price_european_mc(
                   mkt, in.at("k").as_number(), in.at("T").as_number(),
                   [&lv](double k, double t) { return lv.vol(k, t); }, true, s)
            .price;
    }
    if (starts_with(name, "barrier_upout_")) {
        const Market mkt(in.at("s").as_number(), in.at("r").as_number(), in.at("q").as_number());
        McSettings s;
        s.n_paths = static_cast<int>(in.at("n_paths").as_number());
        s.n_steps = static_cast<int>(in.at("n_steps").as_number());
        return price_up_out_call_mc(mkt, in.at("k").as_number(), in.at("b").as_number(),
                                    in.at("T").as_number(), in.at("sigma").as_number(), s, true)
            .price;
    }
    ADD_FAILURE() << "unknown golden case prefix: " << name;
    return 0.0;
}

}  // namespace

TEST_F(GoldenTest, AllCasesWithinTolerance) {
    const auto& cases = root_->at("cases").as_array();
    ASSERT_EQ(cases.size(), 15u);
    for (const auto& cptr : cases) {
        const mini_json::Value& c = *cptr;
        const std::string name = c.at("name").as_string();
        const double tol = c.at("tol").as_number();
        const mini_json::Value& expect = c.at("expect");
        const double want = expect.has("price") ? expect.at("price").as_number()
                                                : expect.at("local_vol").as_number();
        const double got = run_case(c, *bundled_, *flat_);
        EXPECT_NEAR(got, want, tol) << "golden case '" << name << "'";
    }
}
