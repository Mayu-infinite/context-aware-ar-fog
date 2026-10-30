using System;

namespace Cafog.Core
{
    /// <summary>Predicted slowing/stopping along a path (arrays are per GEN step).</summary>
    public sealed class RiskProfile
    {
        public double[] ArcM;
        public double[] Demand;
        public double[] ValueHc;
        public double[] ValuePd;
        public double[] SpeedHc;
        public double[] SpeedPd;
        public double[] Risk;
        public double[] TurnRatio;
        public double[] NarrowRatio;

        public int FirstAbove(double threshold, double maxDistanceM)
        {
            for (int i = 0; i < Risk.Length; i++)
                if (Risk[i] >= threshold && ArcM[i] <= maxDistanceM) return i;
            return -1;
        }

        public double MaxWithin(double maxDistanceM)
        {
            double best = 0.0;
            for (int i = 0; i < Risk.Length; i++)
                if (ArcM[i] <= maxDistanceM) best = Math.Max(best, Risk[i]);
            return best;
        }
    }

    /// <summary>
    /// Go/Explore/No-go action selection (Bhattacharya et al. 2026), port of gen.py:
    /// G = A_G logsig(l_go (dV - th_go)), N = A_N logsig(l_nogo (dV - th_nogo)), v = max(0, G - N),
    /// A_G,cond = A_G (1 - 0.25 F r), A_N,cond = A_N (1 + 0.25 F r), th_nogo = 0.15 F r.
    /// </summary>
    public static class GenModel
    {
        public static double LogSig(double x) => 0.5 * (1.0 + Math.Tanh(0.5 * x));

        public static double Speed(double dv, double demand, PatientProfile p, double fogFactor)
        {
            double severity = fogFactor * demand;
            double aGo = p.goAmplitude * Math.Max(0.0, 1.0 - p.goScaling * severity);
            double aNogo = p.nogoAmplitude * (1.0 + p.nogoScaling * severity);
            double thNogo = p.nogoThresholdShift * severity;
            double go = aGo * LogSig(p.goSlope * (dv - p.goThreshold));
            double nogo = aNogo * LogSig(p.nogoSlope * (dv - thNogo));
            return Math.Max(0.0, go - nogo);
        }

        public static RiskProfile Assess(PredictedPath path, PatientProfile profile, ValueConfig cfg)
        {
            double step = profile.genStepM;
            int m = (int)Math.Floor(path.LengthM / step + 1e-9) + 1;
            double[] arc;
            if (m < 2) arc = new[] { 0.0, path.LengthM };
            else
            {
                arc = new double[m];
                for (int i = 0; i < m; i++) arc[i] = i * step;
            }
            m = arc.Length;

            var costSrc = new double[path.Count];
            for (int i = 0; i < path.Count; i++) costSrc[i] = double.IsInfinity(path.Cost[i]) ? 50.0 : path.Cost[i];
            var cost = new double[m];
            var demand = new double[m];
            var turn = new double[m];
            var narrow = new double[m];
            for (int i = 0; i < m; i++)
            {
                cost[i] = PredictedPath.Interp(arc[i], path.ArcM, costSrc);
                turn[i] = PredictedPath.Interp(arc[i], path.ArcM, path.TurnRatio);
                narrow[i] = PredictedPath.Interp(arc[i], path.ArcM, path.NarrowRatio);
                demand[i] = Math.Max(turn[i], narrow[i]);
            }

            // Cost-to-go along the path, anchored at the far end so it is monotone by construction.
            double endCtg = path.CostToGo[path.Count - 1];
            if (double.IsInfinity(endCtg)) endCtg = 0.0;
            var ctg = new double[m];
            ctg[m - 1] = endCtg;
            for (int i = m - 2; i >= 0; i--) ctg[i] = ctg[i + 1] + 0.5 * (cost[i] + cost[i + 1]) * (arc[i + 1] - arc[i]);

            double logGamma = Math.Log(cfg.discountPerM);
            double kappa = ValueField.ValueCanyonKappa(profile);
            var r = new RiskProfile
            {
                ArcM = new double[m - 1],
                Demand = new double[m - 1],
                ValueHc = new double[m - 1],
                ValuePd = new double[m - 1],
                SpeedHc = new double[m - 1],
                SpeedPd = new double[m - 1],
                Risk = new double[m - 1],
                TurnRatio = new double[m - 1],
                NarrowRatio = new double[m - 1],
            };
            for (int i = 0; i < m - 1; i++)
            {
                double lnHc0 = logGamma * ctg[i], lnHc1 = logGamma * ctg[i + 1];
                double lnPd0 = lnHc0 - kappa * demand[i], lnPd1 = lnHc1 - kappa * demand[i + 1];
                double unit = -logGamma * Math.Max(arc[i + 1] - arc[i], 1e-6);
                double dvHc = (lnHc1 - lnHc0) / unit;
                double dvPd = (lnPd1 - lnPd0) / unit;
                double local = Math.Max(demand[i], demand[i + 1]);
                double vHc = Speed(dvHc, 0.0, profile, 0.0);
                double vPd = Speed(dvPd, local, profile, profile.FogFactor);

                r.ArcM[i] = arc[i + 1];
                r.Demand[i] = local;
                r.ValueHc[i] = Math.Exp(lnHc1);
                r.ValuePd[i] = Math.Exp(lnPd1);
                r.SpeedHc[i] = vHc;
                r.SpeedPd[i] = vPd;
                r.Risk[i] = Math.Min(Math.Max(1.0 - vPd / Math.Max(vHc, 1e-6), 0.0), 1.0);
                r.TurnRatio[i] = Math.Max(turn[i], turn[i + 1]);
                r.NarrowRatio[i] = Math.Max(narrow[i], narrow[i + 1]);
            }
            return r;
        }

        /// <summary>LIP step time (paper Eq. 25).</summary>
        public static double StepTimeS(double speedMps, PatientProfile p) =>
            Math.Min(Math.Max(p.lipBasePeriodS / Math.Pow(Math.Max(speedMps, 1e-3), 0.36), 0.25), 0.9);

        /// <summary>Turn-compensated foot placement half-width (paper Eq. 26).</summary>
        public static double StepHalfWidth(double dpsi, PatientProfile p) =>
            p.stepHalfWidthM * (1.0 + Math.Abs(Math.Sin(dpsi)));
    }
}
