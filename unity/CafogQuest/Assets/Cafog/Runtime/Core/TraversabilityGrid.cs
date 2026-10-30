using System;

namespace Cafog.Core
{
    public static class CellState
    {
        public const sbyte Free = 0;
        public const sbyte Unknown = 1;
        public const sbyte Occupied = 2;
    }

    /// <summary>Per-cell point counts for one depth frame, binned against the grid origin given.</summary>
    public sealed class CellCounts
    {
        public readonly int N;
        public readonly int[] Obstacle;
        public readonly int[] Floor;
        public readonly int[] Dropoff;
        public double OriginX;
        public double OriginZ;
        /// <summary>Floor height used to classify the points (NaN if unknown).</summary>
        public double FloorY = double.NaN;

        public CellCounts(int n)
        {
            N = n;
            Obstacle = new int[n * n];
            Floor = new int[n * n];
            Dropoff = new int[n * n];
        }
    }

    /// <summary>Floor height from a histogram of point heights below the eye (lowest strong peak).</summary>
    public static class FloorEstimator
    {
        public const double BinM = 0.02;
        public const double SearchMinBelowEyeM = 0.5;
        public const double SearchMaxBelowEyeM = 2.3;
        public const int Bins = 90;

        /// <summary>hist[i] counts heights in [eyeY - 2.3 + i*BinM, ... + BinM).</summary>
        public static double? Estimate(int[] hist, double eyeY, double minFraction = 0.02, double peakRatio = 0.3)
        {
            long total = 0;
            for (int i = 0; i < Bins; i++) total += hist[i];
            if (total == 0) return null;

            var smoothed = new double[Bins];
            double peak = 0;
            for (int i = 0; i < Bins; i++)
            {
                double left = i > 0 ? hist[i - 1] : 0;
                double right = i < Bins - 1 ? hist[i + 1] : 0;
                smoothed[i] = 0.25 * left + 0.5 * hist[i] + 0.25 * right;
                peak = Math.Max(peak, smoothed[i]);
            }
            if (peak < minFraction * total) return null;

            int first = 0;
            while (smoothed[first] < peakRatio * peak) first++;
            int end = first;
            while (end + 1 < Bins && smoothed[end + 1] >= smoothed[end]) end++;
            return eyeY - SearchMaxBelowEyeM + (end + 0.5) * BinM;
        }
    }

    /// <summary>
    /// Rolling log-odds occupancy grid on the ground plane, anchored in world coordinates.
    /// Cell (row, col) covers x in [OriginX + col*cs, ...) and z in [OriginZ + row*cs, ...).
    /// Geometry only: no semantic labels.
    /// </summary>
    public sealed class TraversabilityGrid
    {
        public readonly MapConfig Config;
        public readonly int N;
        public readonly double CellSize;
        public float[] LogOdds;
        public bool[] Observed;
        public double OriginX;
        public double OriginZ;
        public double? FloorY;

        double? _lastTime;
        float[] _scratchF;
        bool[] _scratchB;

        public TraversabilityGrid(MapConfig config, Float2 center)
        {
            Config = config;
            N = config.Cells;
            CellSize = config.cellSizeM;
            LogOdds = new float[N * N];
            Observed = new bool[N * N];
            _scratchF = new float[N * N];
            _scratchB = new bool[N * N];
            double half = 0.5 * N * CellSize;
            OriginX = center.X - half;
            OriginZ = center.Z - half;
        }

        public Float2 Center => new Float2(OriginX + 0.5 * N * CellSize, OriginZ + 0.5 * N * CellSize);

        public void WorldToCell(Float2 p, out int row, out int col)
        {
            col = (int)Math.Floor((p.X - OriginX) / CellSize);
            row = (int)Math.Floor((p.Z - OriginZ) / CellSize);
        }

        public Float2 CellCenter(int row, int col) =>
            new Float2(OriginX + (col + 0.5) * CellSize, OriginZ + (row + 0.5) * CellSize);

        public bool InBounds(int row, int col) => row >= 0 && row < N && col >= 0 && col < N;

        /// <summary>Shift by whole cells so the user stays near the centre.</summary>
        public bool Recenter(Float2 user)
        {
            Float2 offset = user - Center;
            if (Math.Max(Math.Abs(offset.X), Math.Abs(offset.Z)) < Config.recenterDistanceM) return false;
            int dc = (int)Math.Round(offset.X / CellSize);
            int dr = (int)Math.Round(offset.Z / CellSize);
            Shift(ref LogOdds, ref _scratchF, dr, dc, 0f);
            Shift(ref Observed, ref _scratchB, dr, dc, false);
            OriginX += dc * CellSize;
            OriginZ += dr * CellSize;
            return true;
        }

        void Shift<T>(ref T[] grid, ref T[] scratch, int dr, int dc, T fill)
        {
            for (int r = 0; r < N; r++)
            {
                int sr = r + dr;
                for (int c = 0; c < N; c++)
                {
                    int sc = c + dc;
                    scratch[r * N + c] = (sr >= 0 && sr < N && sc >= 0 && sc < N) ? grid[sr * N + sc] : fill;
                }
            }
            (grid, scratch) = (scratch, grid);
        }

        public void UpdateFloor(double? estimate)
        {
            if (!estimate.HasValue) return;
            if (!FloorY.HasValue) FloorY = estimate;
            else FloorY = (1 - Config.floorSmoothing) * FloorY.Value + Config.floorSmoothing * estimate.Value;
        }

        public void IntegrateCounts(CellCounts counts, double timestampS)
        {
            var cfg = Config;
            double dt = _lastTime.HasValue ? Math.Max(0.0, timestampS - _lastTime.Value) : 0.0;
            _lastTime = timestampS;
            double decay = dt > 0 && cfg.decayTimeConstantS > 0 ? Math.Exp(-dt / cfg.decayTimeConstantS) : 1.0;

            // Counts may have been binned against an older origin (GPU latency); align by whole cells.
            int dr = (int)Math.Round((counts.OriginZ - OriginZ) / CellSize);
            int dc = (int)Math.Round((counts.OriginX - OriginX) / CellSize);

            for (int r = 0; r < N; r++)
            {
                int sr = r - dr;
                for (int c = 0; c < N; c++)
                {
                    int i = r * N + c;
                    int sc = c - dc;
                    int obstacle = 0, floor = 0, dropoff = 0;
                    if (sr >= 0 && sr < N && sc >= 0 && sc < N)
                    {
                        int j = sr * N + sc;
                        obstacle = counts.Obstacle[j];
                        floor = counts.Floor[j];
                        dropoff = counts.Dropoff[j];
                    }
                    bool hit = obstacle >= cfg.minObstaclePoints || dropoff >= cfg.minObstaclePoints;
                    bool free = floor >= cfg.minFloorPoints && !hit;
                    double v = LogOdds[i];
                    if (hit) v += cfg.logOddsHit;
                    else if (free) v += cfg.logOddsFree;
                    else v *= decay;
                    LogOdds[i] = (float)Math.Min(Math.Max(v, cfg.logOddsMin), cfg.logOddsMax);
                    if (hit || free) Observed[i] = true;
                }
            }
        }

        public void ComputeStates(sbyte[] output)
        {
            var cfg = Config;
            for (int i = 0; i < N * N; i++)
            {
                float v = LogOdds[i];
                if (v >= cfg.occupiedThreshold) output[i] = CellState.Occupied;
                else if (v <= cfg.freeThreshold && Observed[i]) output[i] = CellState.Free;
                else output[i] = CellState.Unknown;
            }
        }
    }
}
