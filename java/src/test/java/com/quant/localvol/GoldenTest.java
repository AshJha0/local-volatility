package com.quant.localvol;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Cross-language golden suite: loads {@code ../data/golden/golden.json} and
 * asserts every case within its stated absolute tolerance, dispatching on the
 * case-name prefix per API_SPEC.md §9.
 */
public class GoldenTest {

    private static final Path DATA = Paths.get("..", "data");
    private static final long SEED = 42L;

    private static List<Object> cases;
    private static ImpliedVolSurface bundled;
    private static ImpliedVolSurface flat;

    @BeforeClass
    public static void load() throws IOException {
        String text = Files.readString(DATA.resolve("golden").resolve("golden.json"));
        cases = Json.asArray(Json.asObject(Json.parse(text)).get("cases"));
        bundled = ImpliedVolSurface.fromCsv(DATA.resolve("implied_surface.csv"));
        flat = ImpliedVolSurface.fromCsv(DATA.resolve("flat_surface.csv"));
    }

    @Test
    public void allFifteenCasesPresent() {
        assertEquals(15, cases.size());
    }

    @Test
    public void allGoldenCasesWithinTolerance() {
        int checked = 0;
        for (Object caseObj : cases) {
            Map<String, Object> c = Json.asObject(caseObj);
            String name = (String) c.get("name");
            Map<String, Object> in = Json.asObject(c.get("inputs"));
            Map<String, Object> expect = Json.asObject(c.get("expect"));
            double tol = Json.asDouble(c.get("tol"));
            double got = dispatch(name, in);
            double want = Json.asDouble(expect.values().iterator().next());
            assertEquals("golden case " + name, want, got, tol);
            checked++;
        }
        assertEquals(15, checked);
    }

    private static double dispatch(String name, Map<String, Object> in) {
        if (name.startsWith("flat_dupire_")) {
            DupireLocalVol lv = new DupireLocalVol(flat);
            return lv.vol(d(in, "k"), d(in, "T"));
        }
        if (name.startsWith("dupire_bundled_")) {
            DupireLocalVol lv = new DupireLocalVol(bundled);
            return lv.vol(d(in, "k"), d(in, "T"));
        }
        if (name.startsWith("pde_flat_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            boolean isCall = d(in, "call") != 0.0;
            return Pde.priceEuropean(mkt, d(in, "k"), d(in, "T"), d(in, "sigma"), isCall,
                    (int) d(in, "num_space"), (int) d(in, "num_time"), 6.0);
        }
        if (name.startsWith("pde_localvol_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            DupireLocalVol lv = new DupireLocalVol(bundled);
            return Pde.priceEuropean(mkt, d(in, "k"), d(in, "T"), lv, d(in, "sigma_ref"), true,
                    (int) d(in, "num_space"), (int) d(in, "num_time"), 6.0);
        }
        if (name.startsWith("american_put_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            return Pde.priceAmericanPut(mkt, d(in, "k"), d(in, "T"), d(in, "sigma"),
                    (int) d(in, "num_space"), (int) d(in, "num_time"), 6.0,
                    Pde.DEFAULT_OMEGA, Pde.DEFAULT_TOL, Pde.DEFAULT_MAX_ITER);
        }
        if (name.startsWith("mc_flat_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            MonteCarlo.McResult res = MonteCarlo.priceEuropean(mkt, d(in, "k"), d(in, "T"),
                    d(in, "sigma"), true, (int) d(in, "n_paths"), (int) d(in, "n_steps"),
                    SEED, true);
            assertTrue("mc_flat stderr must be positive", res.stderr() > 0.0);
            return res.price();
        }
        if (name.startsWith("mc_localvol_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            DupireLocalVol lv = new DupireLocalVol(bundled);
            MonteCarlo.McResult res = MonteCarlo.priceEuropean(mkt, d(in, "k"), d(in, "T"),
                    lv, true, (int) d(in, "n_paths"), (int) d(in, "n_steps"), SEED, true);
            assertTrue("mc_localvol stderr must be positive", res.stderr() > 0.0);
            return res.price();
        }
        if (name.startsWith("barrier_upout_")) {
            Market mkt = new Market(d(in, "s"), d(in, "r"), d(in, "q"));
            MonteCarlo.McResult res = MonteCarlo.priceUpOutCall(mkt, d(in, "k"), d(in, "b"),
                    d(in, "T"), d(in, "sigma"), (int) d(in, "n_paths"), (int) d(in, "n_steps"),
                    SEED, true, true);
            assertTrue("barrier stderr must be positive", res.stderr() > 0.0);
            return res.price();
        }
        throw new AssertionError("unknown golden case prefix: " + name);
    }

    private static double d(Map<String, Object> in, String key) {
        Object v = in.get(key);
        if (v == null) {
            throw new AssertionError("missing golden input key: " + key);
        }
        return Json.asDouble(v);
    }
}
