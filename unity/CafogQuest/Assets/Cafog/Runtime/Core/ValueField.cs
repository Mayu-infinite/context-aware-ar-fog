using System;

namespace Cafog.Core
{
    /// <summary>
    /// Training-free spatial critic (port of src/cafog/spatial/value.py).
    /// V_HC(s) = gamma^D(s), D = cost-weighted geodesic distance to goals ahead (exact, via Dijkstra).
    /// V_PD(s) = V_HC(s) * exp(-kappa * r(s)), r = max(turn ratio, narrowness ratio) on the route.
    /// All buffers are preallocated; Compute() does not allocate.
    /// </summary>
    public sealed class ValueField
    {
        static readonly int[] NeighbourDr = { -1, -1, -1, 0, 0, 1, 1, 1 };
        static readonly int[] NeighbourDc = { -1, 0, 1, -1, 1, -1, 0, 1 };

        public readonly int N;
        public readonly sbyte[] States;
        public readonly float[] ClearanceM;
        public readonly float[] Cost;
        public readonly double[] CostToGo;
        public readonly int[] NextCell;
        public readonly bool[] GoalMask;
        public readonly float[] ValueHc;
        public readonly float[] Demand;
        public readonly float[] ValuePd;
        public int StartCell = -1;

        readonly double[] _goalCost;
        readonly double[] _edtF, _edtD, _edtZ, _sqDist;
        readonly int[] _edtV;
        readonly MinHeap _heap;

        public ValueField(int n)
        {
            N = n;
            int size = n * n;
            States = new sbyte[size];
            ClearanceM = new float[size];
            Cost = new float[size];
            CostToGo = new double[size];
            NextCell = new int[size];
            GoalMask = new bool[size];
            ValueHc = new float[size];
            Demand = new float[size];
            ValuePd = new float[size];
            _goalCost = new double[size];
            _edtF = new double[n];
            _edtD = new double[n];
            _edtZ = new double[n + 1];
            _edtV = new int[n];
            _sqDist = new double[size];
            _heap = new MinHeap(9 * size + 1);
        }

        public bool HasRoute => StartCell >= 0 && !double.IsInfinity(CostToGo[StartCell]);

        public void Compute(TraversabilityGrid grid, Float2 user, Float2 heading, ValueConfig cfg, PatientProfile profile)
        {
            double cs = grid.CellSize;
            grid.ComputeStates(States);
            ComputeClearance(cs);
            ComputeCost(cfg);

            StartCell = -1;
            grid.WorldToCell(user, out int ur, out int uc);
            if (grid.InBounds(ur, uc)) StartCell = NearestTraversable(ur, uc, (int)Math.Round(0.5 / cs));

            ComputeGoals(grid, user, heading, cfg);
            SolveCostToGo(cs);

            double logGamma = Math.Log(cfg.discountPerM);
            double kappa = ValueCanyonKappa(profile);
            ComputeDemand(cfg);
            for (int i = 0; i < N * N; i++)
            {
                double vhc = double.IsInfinity(CostToGo[i]) ? 0.0 : Math.Exp(logGamma * CostToGo[i]);
                ValueHc[i] = (float)vhc;
                ValuePd[i] = (float)(vhc * Math.Exp(-kappa * Demand[i]));
            }
        }

        public static double ValueCanyonKappa(PatientProfile p) => p.valueCanyonGain * p.FogFactor;

        public static double TurnRatio(double turnDeg, ValueConfig cfg)
        {
            double span = 90.0 - cfg.turnDeadbandDeg;
            return Clamp((turnDeg - cfg.turnDeadbandDeg) / span, 0.0, 2.0);
        }

        public static double NarrownessRatio(double widthM, ValueConfig cfg)
        {
            double span = Math.Max(cfg.comfortWidthM - cfg.minWidthM, 1e-6);
            return Clamp((cfg.comfortWidthM - widthM) / span, 0.0, 1.5);
        }

        static double Clamp(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);

        // Distance (m) from each cell centre to the nearest occupied surface (Felzenszwalb EDT).
        void ComputeClearance(double cs)
        {
            const double Inf = 1e20;
            bool any = false;
            for (int i = 0; i < N * N; i++)
            {
                bool occ = States[i] == CellState.Occupied;
                any |= occ;
                _sqDist[i] = occ ? 0.0 : Inf;
            }
            if (!any)
            {
                for (int i = 0; i < N * N; i++) ClearanceM[i] = float.PositiveInfinity;
                return;
            }
            for (int c = 0; c < N; c++)
            {
                for (int r = 0; r < N; r++) _edtF[r] = _sqDist[r * N + c];
                Edt1D();
                for (int r = 0; r < N; r++) _sqDist[r * N + c] = _edtD[r];
            }
            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < N; c++) _edtF[c] = _sqDist[r * N + c];
                Edt1D();
                for (int c = 0; c < N; c++) _sqDist[r * N + c] = _edtD[c];
            }
            for (int i = 0; i < N * N; i++)
                ClearanceM[i] = (float)Math.Max(Math.Sqrt(_sqDist[i]) * cs - 0.5 * cs, 0.0);
        }

        void Edt1D()
        {
            int k = 0;
            _edtV[0] = 0;
            _edtZ[0] = double.NegativeInfinity;
            _edtZ[1] = double.PositiveInfinity;
            for (int q = 1; q < N; q++)
            {
                double s = Intersection(q, _edtV[k]);
                while (s <= _edtZ[k])
                {
                    k--;
                    s = Intersection(q, _edtV[k]);
                }
                k++;
                _edtV[k] = q;
                _edtZ[k] = s;
                _edtZ[k + 1] = double.PositiveInfinity;
            }
            k = 0;
            for (int q = 0; q < N; q++)
            {
                while (_edtZ[k + 1] < q) k++;
                int v = _edtV[k];
                _edtD[q] = (q - v) * (q - v) + _edtF[v];
            }
        }

        double Intersection(int q, int v) => ((_edtF[q] + q * q) - (_edtF[v] + v * v)) / (2.0 * q - 2.0 * v);

        void ComputeCost(ValueConfig cfg)
        {
            for (int i = 0; i < N * N; i++)
            {
                double margin = Math.Min(ClearanceM[i], 10.0) - cfg.bodyRadiusM;
                if (States[i] == CellState.Occupied || margin < 0)
                {
                    Cost[i] = float.PositiveInfinity;
                    continue;
                }
                double c = 1.0 + cfg.narrowCostGain * Math.Exp(-margin / cfg.narrowCostLengthM);
                if (States[i] == CellState.Unknown) c += cfg.unknownCellCost;
                Cost[i] = (float)c;
            }
        }

        bool Traversable(int i) => !float.IsInfinity(Cost[i]);

        int NearestTraversable(int r0, int c0, int radius)
        {
            int best = -1;
            int bestD = int.MaxValue;
            for (int r = Math.Max(0, r0 - radius); r < Math.Min(N, r0 + radius + 1); r++)
            for (int c = Math.Max(0, c0 - radius); c < Math.Min(N, c0 + radius + 1); c++)
            {
                if (!Traversable(r * N + c)) continue;
                int d = (r - r0) * (r - r0) + (c - c0) * (c - c0);
                if (d < bestD)
                {
                    best = r * N + c;
                    bestD = d;
                }
            }
            return best;
        }

        // Goals: free cells at the edge of what is known (frontier or map border) ahead of the user.
        void ComputeGoals(TraversabilityGrid grid, Float2 user, Float2 heading, ValueConfig cfg)
        {
            double cosMax = Math.Cos(cfg.goalMaxAngleDeg * Math.PI / 180.0);
            for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++)
            {
                int i = r * N + c;
                _goalCost[i] = double.PositiveInfinity;
                GoalMask[i] = false;
                if (States[i] != CellState.Free || !Traversable(i)) continue;

                bool frontier = r == 0 || c == 0 || r == N - 1 || c == N - 1;
                for (int dr = -1; dr <= 1 && !frontier; dr++)
                for (int dc = -1; dc <= 1 && !frontier; dc++)
                {
                    int rr = r + dr, cc = c + dc;
                    if (rr >= 0 && rr < N && cc >= 0 && cc < N && States[rr * N + cc] == CellState.Unknown)
                        frontier = true;
                }
                if (!frontier) continue;

                Float2 rel = grid.CellCenter(r, c) - user;
                double dist = rel.Length;
                double cosAngle = rel.Dot(heading) / Math.Max(dist, 1e-6);
                if (dist < cfg.goalMinDistanceM || cosAngle < cosMax) continue;
                double shortfall = Math.Max(0.0, cfg.goalPreferredDistanceM - dist);
                _goalCost[i] = cfg.goalAlignmentCostM * (1.0 - cosAngle) + cfg.goalNearPenalty * shortfall;
                GoalMask[i] = true;
            }
        }

        // Multi-goal Dijkstra on the 8-connected grid without corner cutting.
        void SolveCostToGo(double cs)
        {
            _heap.Clear();
            for (int i = 0; i < N * N; i++)
            {
                NextCell[i] = -1;
                CostToGo[i] = double.PositiveInfinity;
                if (GoalMask[i])
                {
                    CostToGo[i] = _goalCost[i] + 1e-9;
                    _heap.Push(CostToGo[i], i);
                }
            }
            double diag = cs * Math.Sqrt(2.0);
            while (_heap.Count > 0)
            {
                _heap.Pop(out double d, out int u);
                if (d > CostToGo[u]) continue;
                int ur = u / N, uc = u % N;
                for (int k = 0; k < 8; k++)
                {
                    int vr = ur + NeighbourDr[k], vc = uc + NeighbourDc[k];
                    if (vr < 0 || vr >= N || vc < 0 || vc >= N) continue;
                    int v = vr * N + vc;
                    if (!Traversable(v)) continue;
                    bool diagonal = NeighbourDr[k] != 0 && NeighbourDc[k] != 0;
                    if (diagonal && (!Traversable(ur * N + vc) || !Traversable(vr * N + uc))) continue;
                    double w = (diagonal ? diag : cs) * 0.5 * (Cost[u] + Cost[v]);
                    double nd = d + w;
                    if (nd < CostToGo[v])
                    {
                        CostToGo[v] = nd;
                        NextCell[v] = u;
                        _heap.Push(nd, v);
                    }
                }
            }
        }

        int Advance(int cell, int steps)
        {
            for (int s = 0; s < steps; s++)
            {
                int nxt = NextCell[cell];
                if (nxt < 0) break;
                cell = nxt;
            }
            return cell;
        }

        void ComputeDemand(ValueConfig cfg)
        {
            int k = Math.Max(1, cfg.fieldTurnProbeCells);
            for (int i = 0; i < N * N; i++)
            {
                int p1 = Advance(i, k);
                int p2 = Advance(p1, k);
                double d1x = p1 % N - i % N, d1z = p1 / N - i / N;
                double d2x = p2 % N - p1 % N, d2z = p2 / N - p1 / N;
                double n1 = Math.Sqrt(d1x * d1x + d1z * d1z), n2 = Math.Sqrt(d2x * d2x + d2z * d2z);
                double turn = 0.0;
                if (n1 > 0 && n2 > 0)
                {
                    double cosT = Clamp((d1x * d2x + d1z * d2z) / (n1 * n2), -1.0, 1.0);
                    turn = TurnRatio(Math.Acos(cosT) * 180.0 / Math.PI, cfg);
                }
                else turn = TurnRatio(0.0, cfg);
                double narrow = NarrownessRatio(2.0 * Math.Min(ClearanceM[i], 10.0), cfg);
                Demand[i] = (float)Math.Max(turn, narrow);
            }
        }
    }

    /// <summary>Binary min-heap with lazy deletion for Dijkstra.</summary>
    sealed class MinHeap
    {
        readonly double[] _keys;
        readonly int[] _items;
        public int Count { get; private set; }

        public MinHeap(int capacity)
        {
            _keys = new double[capacity];
            _items = new int[capacity];
        }

        public void Clear() => Count = 0;

        public void Push(double key, int item)
        {
            int i = Count++;
            while (i > 0)
            {
                int parent = (i - 1) >> 1;
                if (_keys[parent] <= key) break;
                _keys[i] = _keys[parent];
                _items[i] = _items[parent];
                i = parent;
            }
            _keys[i] = key;
            _items[i] = item;
        }

        public void Pop(out double key, out int item)
        {
            key = _keys[0];
            item = _items[0];
            Count--;
            if (Count == 0) return;
            double lastKey = _keys[Count];
            int lastItem = _items[Count];
            int i = 0;
            while (true)
            {
                int child = 2 * i + 1;
                if (child >= Count) break;
                if (child + 1 < Count && _keys[child + 1] < _keys[child]) child++;
                if (_keys[child] >= lastKey) break;
                _keys[i] = _keys[child];
                _items[i] = _items[child];
                i = child;
            }
            _keys[i] = lastKey;
            _items[i] = lastItem;
        }
    }
}
