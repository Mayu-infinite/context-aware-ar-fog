using System;

namespace Cafog.Core
{
    // Field names mirror src/cafog/spatial/config.py (snake_case -> camelCase) so a clinician
    // profile exported by scripts/export_unity_config.py loads with JsonUtility.

    [Serializable]
    public sealed class MapConfig
    {
        public double cellSizeM = 0.10;
        public double sizeM = 10;
        public double recenterDistanceM = 2;
        public double maxRangeM = 4.5;
        public double floorToleranceM = 0.05;
        public double obstacleMinHeightM = 0.08;
        public double obstacleMaxHeightM = 1.90;
        public double dropoffDepthM = 0.12;
        public int minObstaclePoints = 2;
        public int minFloorPoints = 1;
        public double logOddsHit = 0.9;
        public double logOddsFree = -0.45;
        public double logOddsMin = -2;
        public double logOddsMax = 3;
        public double occupiedThreshold = 0.6;
        public double freeThreshold = -0.3;
        public double decayTimeConstantS = 20;
        public double initialEyeHeightM = 1.6;
        public double floorSmoothing = 0.15;

        public int Cells => (int)Math.Round(sizeM / cellSizeM);
    }

    [Serializable]
    public sealed class ValueConfig
    {
        public double discountPerM = 0.85;
        public double bodyRadiusM = 0.25;
        public double narrowCostGain = 2;
        public double narrowCostLengthM = 0.20;
        public double unknownCellCost = 1;
        public double goalMinDistanceM = 1;
        public double goalPreferredDistanceM = 4.5;
        public double goalNearPenalty = 2;
        public double goalAlignmentCostM = 3;
        public double goalMaxAngleDeg = 135;
        public double lookaheadM = 4;
        public int pathSmoothingWindow = 5;
        public double pathSampleStepM = 0.10;
        public double turnWindowM = 1;
        public double turnDeadbandDeg = 15;
        public double widthRayMaxM = 1.5;
        public double comfortWidthM = 1.1;
        public double minWidthM = 0.6;
        public int fieldTurnProbeCells = 5;
    }

    /// <summary>GEN model parameters (Bhattacharya et al. 2026). deltaLim &lt;= 0 means healthy.</summary>
    [Serializable]
    public sealed class PatientProfile
    {
        public string name = "pd_fog";
        public double deltaLim = 0.001;
        public double deltaLimRef = 0.01;
        public double goAmplitude = 3;
        public double nogoAmplitude = 3;
        public double goSlope = 1;
        public double nogoSlope = -10;
        public double goThreshold = 0;
        public double goScaling = 0.25;
        public double nogoScaling = 0.25;
        public double nogoThresholdShift = 0.15;
        public double valueCanyonGain = 0.08;
        public double genStepM = 0.5;
        public double cueStepLengthM = 0.55;
        public double stepHalfWidthM = 0.10;
        public double cueSpeedMps = 0.8;
        public double lipBasePeriodS = 0.5;

        public double FogFactor => deltaLim > 0 ? deltaLimRef / deltaLim : 0.0;

        public static PatientProfile Preset(string preset)
        {
            switch (preset)
            {
                case "healthy": return new PatientProfile { name = "healthy", deltaLim = 0f };
                case "pd_nonfog": return new PatientProfile { name = "pd_nonfog", deltaLim = 0.008f };
                case "pd_fog": return new PatientProfile { name = "pd_fog", deltaLim = 0.001f };
                default: throw new ArgumentException($"Unknown profile preset '{preset}'");
            }
        }
    }

    [Serializable]
    public sealed class CueConfig
    {
        public double onRisk = 0.6;
        public double offRisk = 0.35;
        public double onHoldS = 0.2;
        public double offHoldS = 1;
        public double minActiveS = 1.5;
        public double triggerDistanceM = 3;
        public double initiationDistanceM = 1.5;
        public double stationarySpeedMps = 0.12;
        public double extendPastHazardM = 1.5;
        public double firstStepOffsetM = 0.35;
        public int maxFootsteps = 8;
        public double replanDeviationM = 0.35;
        public double replanHoldS = 0.5;
        public double footstepMinClearanceM = 0.15;
        public bool audioEnabled = true;
        public bool visualEnabled = true;
    }

    [Serializable]
    public sealed class EngineConfig
    {
        public MapConfig map = new MapConfig();
        public ValueConfig value = new ValueConfig();
        public PatientProfile profile = new PatientProfile();
        public CueConfig cue = new CueConfig();
        public int depthStride = 2;
        public double headingSpeedThresholdMps = 0.25;
        public double headingTimeConstantS = 0.5;
        public double velocityTimeConstantS = 0.3;
    }
}
