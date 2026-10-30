using System;
using System.Collections.Generic;

namespace Cafog.Core
{
    public enum CuePhase { Idle, Arming, Active, Releasing }

    public enum HazardKind { Turn, Narrow, Initiation }

    public readonly struct Footstep
    {
        public readonly Float2 Position;
        /// <summary>Walking direction at the step (unit vector on the ground plane).</summary>
        public readonly Float2 Forward;
        /// <summary>-1 left foot, +1 right foot.</summary>
        public readonly int Side;
        public readonly int Index;

        public Footstep(Float2 position, Float2 forward, int side, int index)
        {
            Position = position;
            Forward = forward;
            Side = side;
            Index = index;
        }
    }

    public sealed class Hazard
    {
        public HazardKind Kind;
        public double DistanceM;
        public double Risk;
        public double TurnDeg;
        public double WidthM;
    }

    public sealed class CueOutput
    {
        public CuePhase Phase;
        public List<Footstep> Footsteps = new List<Footstep>();
        /// <summary>Metronome tempo, or 0 when the auditory cue is off.</summary>
        public double MetronomeBpm;
        public Hazard Hazard;
        public string Reason = "";

        public bool Active => Phase == CuePhase.Active || Phase == CuePhase.Releasing;
    }

    /// <summary>Turns a risk profile into a stable, minimal cue (port of cues.py).</summary>
    public sealed class CueController
    {
        readonly CueConfig _cfg;
        readonly PatientProfile _profile;
        public CuePhase Phase { get; private set; } = CuePhase.Idle;
        List<Footstep> _footsteps = new List<Footstep>();
        double? _since;
        double _activatedAt;
        double? _deviatingSince;
        Hazard _hazard;

        public CueController(CueConfig cfg, PatientProfile profile)
        {
            _cfg = cfg;
            _profile = profile;
        }

        public static List<Footstep> PlanFootsteps(PredictedPath path, double endS, PatientProfile profile, CueConfig cfg, int firstSide = -1)
        {
            var steps = new List<Footstep>();
            double stride = profile.cueStepLengthM;
            endS = Math.Min(endS, path.LengthM);
            int side = firstSide;
            for (int i = 0; i < cfg.maxFootsteps; i++)
            {
                double s = cfg.firstStepOffsetM + i * stride;
                if (s > endS + 1e-9) break;
                int k = Math.Min(Math.Max(PredictedPath.SearchSorted(path.ArcM, s), 0), path.Count - 1);
                if (!path.KnownFree[k] || path.ClearanceM[k] < cfg.footstepMinClearanceM) break;
                Float2 tangent = path.TangentAt(s);
                Float2 ahead = path.TangentAt(s + stride);
                double dpsi = Math.Atan2(tangent.Cross(ahead), tangent.Dot(ahead));
                double halfWidth = GenModel.StepHalfWidth(dpsi, profile);
                Float2 pos = path.PointAt(s) + tangent.RightOf() * (side * halfWidth);
                steps.Add(new Footstep(pos, tangent, side, i));
                side = -side;
            }
            return steps;
        }

        static double Deviation(List<Footstep> steps, PredictedPath path)
        {
            if (steps.Count == 0) return double.PositiveInfinity;
            double sum = 0;
            foreach (var step in steps)
            {
                double best = double.PositiveInfinity;
                foreach (var p in path.Points) best = Math.Min(best, (step.Position - p).Length);
                sum += best;
            }
            return sum / steps.Count;
        }

        Hazard Detect(RiskProfile risk, PredictedPath path, double speed)
        {
            bool stationary = speed < _cfg.stationarySpeedMps;
            double reach = stationary ? _cfg.initiationDistanceM : _cfg.triggerDistanceM;
            int k = risk.FirstAbove(_cfg.onRisk, reach);
            if (k < 0) return null;
            double s = risk.ArcM[k];
            double end = s + _cfg.extendPastHazardM;

            double maxRisk = 0, maxTurnRatio = 0, maxNarrowRatio = 0;
            for (int i = 0; i < risk.ArcM.Length; i++)
            {
                if (risk.ArcM[i] < s || risk.ArcM[i] > end) continue;
                maxRisk = Math.Max(maxRisk, risk.Risk[i]);
                maxTurnRatio = Math.Max(maxTurnRatio, risk.TurnRatio[i]);
                maxNarrowRatio = Math.Max(maxNarrowRatio, risk.NarrowRatio[i]);
            }
            double turnDeg = 0, width = double.PositiveInfinity;
            for (int i = 0; i < path.Count; i++)
            {
                if (path.ArcM[i] < s - 0.5 || path.ArcM[i] > end) continue;
                turnDeg = Math.Max(turnDeg, path.TurnDeg[i]);
                width = Math.Min(width, path.WidthM[i]);
            }
            var kind = maxTurnRatio >= maxNarrowRatio ? HazardKind.Turn : HazardKind.Narrow;
            if (stationary) kind = HazardKind.Initiation;
            return new Hazard { Kind = kind, DistanceM = s, Risk = maxRisk, TurnDeg = turnDeg, WidthM = width };
        }

        public CueOutput Update(RiskProfile risk, PredictedPath path, Float2 user, double speedMps, double t)
        {
            if (risk == null || path == null) return Release(t, "no traversable route ahead", force: true);

            Hazard hazard = Detect(risk, path, speedMps);
            double level = risk.MaxWithin(_cfg.triggerDistanceM + _cfg.extendPastHazardM);

            if (Phase == CuePhase.Idle || Phase == CuePhase.Arming)
            {
                if (hazard == null)
                {
                    Phase = CuePhase.Idle;
                    _since = null;
                    return new CueOutput { Phase = Phase, Reason = "spatial demand within tolerance" };
                }
                if (Phase == CuePhase.Idle)
                {
                    Phase = CuePhase.Arming;
                    _since = t;
                }
                if (t - _since.Value >= _cfg.onHoldS) Activate(path, hazard, t);
                return Output($"{hazard.Kind} risk {hazard.Risk:F2}");
            }

            _hazard = hazard ?? _hazard;
            Prune(user);
            MaybeReplan(path, t);

            if (level <= _cfg.offRisk && t - _activatedAt >= _cfg.minActiveS)
            {
                if (Phase != CuePhase.Releasing)
                {
                    Phase = CuePhase.Releasing;
                    _since = t;
                }
                else if (t - _since.Value >= _cfg.offHoldS) return Release(t, "demand resolved");
            }
            else if (Phase == CuePhase.Releasing) Phase = CuePhase.Active;

            if (_footsteps.Count == 0) _footsteps = Plan(path);
            return Output("guiding through high-demand space");
        }

        void Activate(PredictedPath path, Hazard hazard, double t)
        {
            Phase = CuePhase.Active;
            _activatedAt = t;
            _hazard = hazard;
            _footsteps = Plan(path);
        }

        List<Footstep> Plan(PredictedPath path)
        {
            double end = (_hazard?.DistanceM ?? 0.0) + _cfg.extendPastHazardM;
            return PlanFootsteps(path, end, _profile, _cfg);
        }

        void Prune(Float2 user)
        {
            _footsteps.RemoveAll(step => (step.Position - user).Dot(step.Forward) <= 0.1);
        }

        void MaybeReplan(PredictedPath path, double t)
        {
            if (Deviation(_footsteps, path) <= _cfg.replanDeviationM)
            {
                _deviatingSince = null;
                return;
            }
            if (!_deviatingSince.HasValue) _deviatingSince = t;
            else if (t - _deviatingSince.Value >= _cfg.replanHoldS)
            {
                _footsteps = Plan(path);
                _deviatingSince = null;
            }
        }

        CueOutput Release(double t, string reason, bool force = false)
        {
            if (force && Phase == CuePhase.Active && t - _activatedAt < _cfg.minActiveS)
                return Output("holding cue: route temporarily lost");
            Phase = CuePhase.Idle;
            _footsteps = new List<Footstep>();
            _since = null;
            _hazard = null;
            return new CueOutput { Phase = Phase, Reason = reason };
        }

        CueOutput Output(string reason)
        {
            bool active = Phase == CuePhase.Active || Phase == CuePhase.Releasing;
            var output = new CueOutput { Phase = Phase, Reason = reason };
            if (active && _cfg.audioEnabled) output.MetronomeBpm = 60.0 / GenModel.StepTimeS(_profile.cueSpeedMps, _profile);
            if (active && _cfg.visualEnabled) output.Footsteps = new List<Footstep>(_footsteps);
            if (active) output.Hazard = _hazard;
            return output;
        }
    }
}
