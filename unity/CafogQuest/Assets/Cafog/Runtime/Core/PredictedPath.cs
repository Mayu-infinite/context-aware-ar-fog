using System;
using System.Collections.Generic;

namespace Cafog.Core
{
    /// <summary>Arc-length-resampled route the user is expected to walk next (port of path.py).</summary>
    public sealed class PredictedPath
    {
        public const double WidthRayStepM = 0.05;
        public const double HeadingLeadM = 0.3;

        public Float2[] Points;
        public double[] ArcM;
        public Float2[] Tangents;
        public double[] TurnDeg;
        public double[] TurnRatio;
        public double[] WidthM;
        public double[] NarrowRatio;
        public double[] CostToGo;
        public double[] Cost;
        public bool[] KnownFree;
        public double[] ClearanceM;

        public int Count => ArcM.Length;
        public double LengthM => ArcM.Length > 0 ? ArcM[ArcM.Length - 1] : 0.0;

        public Float2 PointAt(double s)
        {
            int i = SearchSorted(ArcM, s);
            if (i <= 0) return Points[0];
            if (i >= Count) return Points[Count - 1];
            double t = (s - ArcM[i - 1]) / Math.Max(ArcM[i] - ArcM[i - 1], 1e-12);
            return Points[i - 1] + (Points[i] - Points[i - 1]) * t;
        }

        public Float2 TangentAt(double s) => Tangents[Math.Min(Math.Max(SearchSorted(ArcM, s), 0), Count - 1)];

        /// <summary>First index i with a[i] &gt;= x (numpy searchsorted, side='left').</summary>
        public static int SearchSorted(double[] a, double x)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (a[mid] < x) lo = mid + 1;
                else hi = mid;
            }
            return lo;
        }

        /// <summary>numpy.interp: linear interpolation, clamped to the end values.</summary>
        public static double Interp(double x, double[] xp, double[] fp)
        {
            int n = xp.Length;
            if (x <= xp[0]) return fp[0];
            if (x >= xp[n - 1]) return fp[n - 1];
            int i = SearchSorted(xp, x);
            if (xp[i] == x) return fp[i];
            double t = (x - xp[i - 1]) / (xp[i] - xp[i - 1]);
            return fp[i - 1] + t * (fp[i] - fp[i - 1]);
        }

        public static PredictedPath Predict(TraversabilityGrid grid, ValueField field, Float2 user, Float2 heading, ValueConfig cfg)
        {
            double cs = grid.CellSize;
            int n = field.N;
            if (!field.HasRoute) return null;

            var raw = new List<Float2> { user - heading * HeadingLeadM, user };
            int cur = field.StartCell;
            int maxCells = (int)(4 * cfg.lookaheadM / cs) + 10;
            for (int k = 0; k < maxCells; k++)
            {
                int nxt = field.NextCell[cur];
                if (nxt < 0) break;
                cur = nxt;
                raw.Add(grid.CellCenter(cur / n, cur % n));
            }
            if (raw.Count < 3) return null;

            var smoothed = Smooth(raw, cfg.pathSmoothingWindow);
            Resample(smoothed, cfg.pathSampleStepM, out var points, out var arc);
            if (points.Count < 3) return null;

            int m = points.Count;
            var tangents = new Float2[m];
            var headings = new double[m];
            for (int i = 0; i < m; i++)
            {
                Float2 g = i == 0 ? points[1] - points[0]
                    : i == m - 1 ? points[m - 1] - points[m - 2]
                    : (points[i + 1] - points[i - 1]) * 0.5;
                tangents[i] = g.Normalized();
                headings[i] = Math.Atan2(tangents[i].Z, tangents[i].X);
            }
            var turn = WindowedTurnDeg(headings, arc, cfg.turnWindowM);

            // Drop the synthetic lead-in so arc length 0 is the user's position.
            int start = 0;
            double best = double.PositiveInfinity;
            for (int i = 0; i < m; i++)
            {
                double d = Math.Abs(arc[i] - HeadingLeadM);
                if (d < best)
                {
                    best = d;
                    start = i;
                }
            }
            double arc0 = arc[start];
            int end = start;
            while (end < m && arc[end] - arc0 <= cfg.lookaheadM + 1e-9) end++;
            int count = end - start;
            if (count < 2) return null;

            var path = new PredictedPath
            {
                Points = new Float2[count],
                ArcM = new double[count],
                Tangents = new Float2[count],
                TurnDeg = new double[count],
                TurnRatio = new double[count],
                WidthM = new double[count],
                NarrowRatio = new double[count],
                CostToGo = new double[count],
                Cost = new double[count],
                KnownFree = new bool[count],
                ClearanceM = new double[count],
            };
            for (int j = 0; j < count; j++)
            {
                int i = start + j;
                Float2 p = points[i];
                path.Points[j] = p;
                path.ArcM[j] = arc[i] - arc0;
                path.Tangents[j] = tangents[i];
                path.TurnDeg[j] = turn[i];
                path.TurnRatio[j] = ValueField.TurnRatio(turn[i], cfg);
                // Rays stop at the first occupied cell, which on average starts half a cell early on each side.
                path.WidthM[j] = LateralWidth(grid, field.States, p, tangents[i], cfg.widthRayMaxM) + cs;
                path.NarrowRatio[j] = ValueField.NarrownessRatio(path.WidthM[j], cfg);

                grid.WorldToCell(p, out int r, out int c);
                bool inside = grid.InBounds(r, c);
                int idx = r * n + c;
                path.CostToGo[j] = inside ? field.CostToGo[idx] : double.PositiveInfinity;
                path.Cost[j] = inside ? field.Cost[idx] : double.PositiveInfinity;
                path.KnownFree[j] = inside && field.States[idx] == CellState.Free;
                path.ClearanceM[j] = inside ? field.ClearanceM[idx] : 0.0;
            }
            return path;
        }

        static List<Float2> Smooth(List<Float2> pts, int window)
        {
            if (pts.Count < 3 || window < 2) return pts;
            int half = window / 2;
            int n = pts.Count;
            var output = new List<Float2>(n);
            for (int i = 0; i < n; i++)
            {
                double sx = 0, sz = 0;
                for (int k = -half; k <= half; k++)
                {
                    Float2 p = pts[Math.Min(Math.Max(i + k, 0), n - 1)];
                    sx += p.X;
                    sz += p.Z;
                }
                output.Add(new Float2(sx / (2 * half + 1), sz / (2 * half + 1)));
            }
            output[0] = pts[0];
            output[n - 1] = pts[n - 1];
            return output;
        }

        static void Resample(List<Float2> pts, double step, out List<Float2> output, out double[] samples)
        {
            var kept = new List<Float2> { pts[0] };
            for (int i = 1; i < pts.Count; i++)
                if ((pts[i] - pts[i - 1]).Length > 1e-6) kept.Add(pts[i]);
            output = new List<Float2>();
            if (kept.Count < 2)
            {
                output.AddRange(kept);
                samples = new double[kept.Count];
                return;
            }
            var arc = new double[kept.Count];
            for (int i = 1; i < kept.Count; i++) arc[i] = arc[i - 1] + (kept[i] - kept[i - 1]).Length;
            double total = arc[kept.Count - 1];
            var s = new List<double>();
            for (int k = 0; k * step <= total + 1e-9; k++) s.Add(k * step);
            if (s[s.Count - 1] < total - 1e-6) s.Add(total);
            samples = s.ToArray();

            int seg = 1;
            foreach (double x in samples)
            {
                while (seg < kept.Count - 1 && arc[seg] < x) seg++;
                double t = Math.Min(Math.Max((x - arc[seg - 1]) / Math.Max(arc[seg] - arc[seg - 1], 1e-12), 0.0), 1.0);
                output.Add(kept[seg - 1] + (kept[seg] - kept[seg - 1]) * t);
            }
        }

        static double[] WindowedTurnDeg(double[] headings, double[] arc, double windowM)
        {
            int n = headings.Length;
            var unwrapped = new double[n];
            unwrapped[0] = headings[0];
            for (int i = 1; i < n; i++)
            {
                double d = headings[i] - headings[i - 1];
                d -= 2 * Math.PI * Math.Round(d / (2 * Math.PI));
                unwrapped[i] = unwrapped[i - 1] + d;
            }
            var turn = new double[n];
            for (int i = 0; i < n; i++)
            {
                double before = Interp(arc[i] - 0.5 * windowM, arc, unwrapped);
                double after = Interp(arc[i] + 0.5 * windowM, arc, unwrapped);
                turn[i] = Math.Abs(after - before) * 180.0 / Math.PI;
            }
            return turn;
        }

        static double LateralWidth(TraversabilityGrid grid, sbyte[] states, Float2 p, Float2 tangent, double maxM)
        {
            Float2 lateral = tangent.RightOf();
            double total = 0;
            int n = grid.N;
            for (int side = 1; side >= -1; side -= 2)
            {
                double free = maxM;
                for (int k = 1; k * WidthRayStepM <= maxM + 1e-9; k++)
                {
                    double d = k * WidthRayStepM;
                    grid.WorldToCell(p + lateral * (side * d), out int r, out int c);
                    if (grid.InBounds(r, c) && states[r * n + c] == CellState.Occupied)
                    {
                        free = d - WidthRayStepM;
                        break;
                    }
                }
                total += free;
            }
            return total;
        }
    }
}
