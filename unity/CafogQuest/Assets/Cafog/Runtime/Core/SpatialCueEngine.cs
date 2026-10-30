using System;
using System.Diagnostics;

namespace Cafog.Core
{
    /// <summary>Smoothed walking direction and speed from the head pose.</summary>
    public sealed class EgoMotion
    {
        readonly double _speedThreshold, _headingTau, _velocityTau;
        Float2? _position;
        double? _t;
        public Float2 Velocity;
        public Float2 Heading = new Float2(0, 1);

        public EgoMotion(double speedThreshold, double headingTau, double velocityTau)
        {
            _speedThreshold = speedThreshold;
            _headingTau = headingTau;
            _velocityTau = velocityTau;
        }

        public double Speed => Velocity.Length;

        public void Update(Float2 position, Float2? gaze, double t)
        {
            if (!_position.HasValue || !_t.HasValue)
            {
                _position = position;
                _t = t;
                if (gaze.HasValue) Heading = gaze.Value;
                return;
            }
            double dt = t - _t.Value;
            if (dt <= 0) return;
            double aV = 1.0 - Math.Exp(-dt / _velocityTau);
            Velocity = Velocity * (1 - aV) + (position - _position.Value) * (aV / dt);
            _position = position;
            _t = t;

            Float2? target = Speed >= _speedThreshold ? Velocity * (1.0 / Speed) : gaze;
            if (!target.HasValue) return;
            double aH = 1.0 - Math.Exp(-dt / _headingTau);
            Float2 blended = Heading * (1 - aH) + target.Value * aH;
            if (blended.Length > 1e-6) Heading = blended.Normalized();
        }
    }

    public sealed class EngineResult
    {
        public double TimestampS;
        public Float2 User;
        public Float2 Heading;
        public double SpeedMps;
        public double FloorY;
        public PredictedPath Path;
        public RiskProfile Risk;
        public CueOutput Cue;
        public double TotalMs;
    }

    /// <summary>
    /// Binned depth + head pose in, value field + footstep cue out. No labels, no training.
    /// Port of src/cafog/spatial/engine.py (process_counts entry point). Not thread-safe:
    /// call ProcessCounts from one thread; read OriginForNextFrame from any thread.
    /// </summary>
    public sealed class SpatialCueEngine
    {
        public readonly EngineConfig Config;
        public TraversabilityGrid Grid { get; private set; }
        public ValueField Field { get; private set; }
        /// <summary>Held while Field is being rewritten; take it to read the heatmap consistently.</summary>
        public readonly object FieldLock = new object();

        readonly EgoMotion _ego;
        readonly CueController _cues;
        readonly Stopwatch _watch = new Stopwatch();
        double _originX, _originZ;
        volatile bool _hasOrigin;

        public SpatialCueEngine(EngineConfig config)
        {
            Config = config;
            _ego = new EgoMotion(config.headingSpeedThresholdMps, config.headingTimeConstantS, config.velocityTimeConstantS);
            _cues = new CueController(config.cue, config.profile);
        }

        /// <summary>Grid origin the GPU should bin the next frame against (false before the first frame).</summary>
        public bool TryGetOrigin(out double originX, out double originZ)
        {
            lock (FieldLock)
            {
                originX = _originX;
                originZ = _originZ;
            }
            return _hasOrigin;
        }

        /// <summary>Origin a not-yet-created grid centred on the user would have.</summary>
        public double InitialOriginX(Float2 user) => user.X - 0.5 * Config.map.Cells * Config.map.cellSizeM;
        public double InitialOriginZ(Float2 user) => user.Z - 0.5 * Config.map.Cells * Config.map.cellSizeM;

        public EngineResult ProcessCounts(CellCounts counts, Float2 user, Float2? gaze, double timestampS)
        {
            _watch.Restart();
            if (Grid == null)
            {
                Grid = new TraversabilityGrid(Config.map, user);
                Field = new ValueField(Grid.N);
            }
            if (!double.IsNaN(counts.FloorY)) Grid.FloorY = counts.FloorY;
            Grid.Recenter(user);
            Grid.IntegrateCounts(counts, timestampS);

            _ego.Update(user, gaze, timestampS);
            Float2 heading = _ego.Heading;
            lock (FieldLock)
            {
                Field.Compute(Grid, user, heading, Config.value, Config.profile);
                _originX = Grid.OriginX;
                _originZ = Grid.OriginZ;
                _hasOrigin = true;
            }
            PredictedPath path = PredictedPath.Predict(Grid, Field, user, heading, Config.value);
            RiskProfile risk = path != null ? GenModel.Assess(path, Config.profile, Config.value) : null;
            CueOutput cue = _cues.Update(risk, path, user, _ego.Speed, timestampS);
            _watch.Stop();

            return new EngineResult
            {
                TimestampS = timestampS,
                User = user,
                Heading = heading,
                SpeedMps = _ego.Speed,
                FloorY = Grid.FloorY ?? double.NaN,
                Path = path,
                Risk = risk,
                Cue = cue,
                TotalMs = _watch.Elapsed.TotalMilliseconds,
            };
        }
    }
}
