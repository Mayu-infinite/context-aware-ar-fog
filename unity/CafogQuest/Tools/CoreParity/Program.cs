using System;
using System.Diagnostics;
using System.IO;
using Cafog.Core;

// Replays traces written by scripts/export_parity_trace.py and compares cue decisions.
static class Program
{
    static readonly bool Verbose = Environment.GetEnvironmentVariable("PARITY_VERBOSE") == "1";

    static int Main(string[] args)
    {
        string dir = args.Length > 0 ? args[0] : "experiments/exp007_csharp_parity/results";
        Float2.Handedness = -1.0;
        int failures = 0;
        foreach (string file in Directory.GetFiles(dir, "*.bin"))
            failures += Replay(file);
        Console.WriteLine(failures == 0 ? "PARITY OK" : $"PARITY FAILED in {failures} trace(s)");
        return failures == 0 ? 0 : 1;
    }

    static int Replay(string file)
    {
        using var reader = new BinaryReader(File.OpenRead(file));
        int n = reader.ReadInt32();
        int frames = reader.ReadInt32();
        var engine = new SpatialCueEngine(new EngineConfig { profile = PatientProfile.Preset("pd_fog") });
        var counts = new CellCounts(n);

        int phaseMismatch = 0, stepsMismatch = 0, active = 0, pyActive = 0;
        double riskErr = 0, footErr = 0, maxMs = 0, totalMs = 0;
        int footCompared = 0, riskCompared = 0;
        var watch = new Stopwatch();
        for (int f = 0; f < frames; f++)
        {
            double t = reader.ReadDouble(), ux = reader.ReadDouble(), uz = reader.ReadDouble();
            double gx = reader.ReadDouble(), gz = reader.ReadDouble();
            counts.OriginX = reader.ReadDouble();
            counts.OriginZ = reader.ReadDouble();
            counts.FloorY = reader.ReadDouble();
            foreach (int[] grid in new[] { counts.Obstacle, counts.Floor, counts.Dropoff })
                for (int i = 0; i < n * n; i++) grid[i] = reader.ReadInt32();
            int pyPhase = reader.ReadInt32(), pySteps = reader.ReadInt32();
            double pyRisk = reader.ReadDouble(), pyLen = reader.ReadDouble(), pyHazard = reader.ReadDouble();
            double pyFx = reader.ReadDouble(), pyFz = reader.ReadDouble();

            Float2? gaze = double.IsNaN(gx) ? (Float2?)null : new Float2(gx, gz);
            watch.Restart();
            EngineResult r = engine.ProcessCounts(counts, new Float2(ux, uz), gaze, t);
            watch.Stop();
            maxMs = Math.Max(maxMs, watch.Elapsed.TotalMilliseconds);
            totalMs += watch.Elapsed.TotalMilliseconds;

            if ((int)r.Cue.Phase != pyPhase) phaseMismatch++;
            if (r.Cue.Footsteps.Count != pySteps) stepsMismatch++;
            if (r.Cue.Active) active++;
            if (pyPhase >= 2) pyActive++;
            if (Verbose && ((int)r.Cue.Phase != pyPhase || (r.Risk != null && Math.Abs(r.Risk.MaxWithin(3.0) - pyRisk) > 0.1)))
                Console.WriteLine(
                    $"  f{f} z={uz:F2} phase C#={r.Cue.Phase} py={pyPhase} risk C#={r.Risk?.MaxWithin(3.0):F2} py={pyRisk:F2} " +
                    $"len C#={r.Path?.LengthM:F2} py={pyLen:F2} hazard C#={r.Cue.Hazard?.DistanceM:F2} py={pyHazard:F2}");
            if (r.Risk != null && !double.IsNaN(pyRisk))
            {
                riskErr = Math.Max(riskErr, Math.Abs(r.Risk.MaxWithin(3.0) - pyRisk));
                riskCompared++;
            }
            if (r.Cue.Footsteps.Count > 0 && !double.IsNaN(pyFx))
            {
                if (Verbose && (r.Cue.Footsteps[0].Position - new Float2(pyFx, pyFz)).Length > 0.05)
                    Console.WriteLine(
                        $"  f{f} user=({ux:F2},{uz:F2}) first step C#={r.Cue.Footsteps[0].Position} py=({pyFx:F2},{pyFz:F2}) " +
                        $"steps C#={r.Cue.Footsteps.Count} py={pySteps}");
                footErr = Math.Max(footErr, (r.Cue.Footsteps[0].Position - new Float2(pyFx, pyFz)).Length);
                footCompared++;
            }
        }
        bool ok = phaseMismatch == 0 && stepsMismatch == 0 && riskErr < 1e-6 && footErr < 1e-3;
        Console.WriteLine(
            $"{Path.GetFileNameWithoutExtension(file),-12} frames={frames} active C#={active} py={pyActive} " +
            $"phase-mismatch={phaseMismatch} steps-mismatch={stepsMismatch} " +
            $"max|risk diff|={riskErr:F3} ({riskCompared}) max first-step diff={footErr:F3} m ({footCompared}) " +
            $"latency mean={totalMs / frames:F2} ms max={maxMs:F2} ms {(ok ? "OK" : "MISMATCH")}");
        return ok ? 0 : 1;
    }
}
