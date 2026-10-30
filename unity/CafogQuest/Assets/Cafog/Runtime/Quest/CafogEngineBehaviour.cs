using System;
using System.Threading;
using Cafog.Core;
using UnityEngine;

namespace Cafog.Quest
{
    /// <summary>
    /// Runs the label-free cue engine on the headset: GPU depth binning -> background-thread
    /// value field, GEN risk and cue controller -> latest result for the cue renderers.
    /// </summary>
    public sealed class CafogEngineBehaviour : MonoBehaviour
    {
        public enum ProfilePreset { Healthy, PdNonFog, PdFog, Custom }

        [SerializeField] QuestDepthGridSource depthSource;
        [Tooltip("Centre-eye transform of the camera rig (defaults to Camera.main).")]
        [SerializeField] Transform head;
        [SerializeField] ProfilePreset preset = ProfilePreset.PdFog;
        [Tooltip("Optional JSON exported by scripts/export_unity_config.py; overrides the fields below.")]
        [SerializeField] TextAsset configJson;
        [SerializeField] EngineConfig config = new EngineConfig();
        [Tooltip("Run planning off the main thread (recommended on device).")]
        [SerializeField] bool useWorkerThread = true;

        SpatialCueEngine _engine;
        Thread _worker;
        readonly AutoResetEvent _wake = new AutoResetEvent(false);
        volatile bool _running;
        Job _mailbox;
        readonly object _mailboxLock = new object();
        EngineResult _latest;
        int _resultVersion;

        public SpatialCueEngine Engine => _engine;
        public EngineConfig Config => config;
        /// <summary>Latest engine output (null until the first depth frame is processed).</summary>
        public EngineResult Latest => Volatile.Read(ref _latest);
        /// <summary>Increments every time a new result is published.</summary>
        public int ResultVersion => Volatile.Read(ref _resultVersion);

        sealed class Job
        {
            public CellCounts Counts;
            public Float2 User;
            public Float2? Gaze;
            public double Time;
        }

        void Awake()
        {
            if (head == null && Camera.main != null) head = Camera.main.transform;
            if (configJson != null) JsonUtility.FromJsonOverwrite(configJson.text, config);
            else if (preset != ProfilePreset.Custom) config.profile = PatientProfile.Preset(PresetName(preset));
            Rebuild();
        }

        static string PresetName(ProfilePreset p) =>
            p == ProfilePreset.Healthy ? "healthy" : p == ProfilePreset.PdNonFog ? "pd_nonfog" : "pd_fog";

        /// <summary>Apply a clinician/companion-app profile at runtime (JSON with EngineConfig fields).</summary>
        public void ApplyConfigJson(string json)
        {
            var updated = new EngineConfig();
            JsonUtility.FromJsonOverwrite(json, updated);
            config = updated;
            Rebuild();
        }

        void Rebuild()
        {
            StopWorker();
            _engine = new SpatialCueEngine(config);
            Volatile.Write(ref _latest, null);
            if (useWorkerThread)
            {
                _running = true;
                _worker = new Thread(WorkerLoop) { IsBackground = true, Name = "Cafog engine" };
                _worker.Start();
            }
        }

        void OnDestroy() => StopWorker();

        void StopWorker()
        {
            if (_worker == null) return;
            _running = false;
            _wake.Set();
            _worker.Join(200);
            _worker = null;
        }

        void Update()
        {
            if (head == null || depthSource == null || depthSource.Busy) return;
            Vector3 p = head.position;
            Vector3 f = head.forward;
            var user = new Float2(p.x, p.z);
            var planar = new Float2(f.x, f.z);
            Float2? gaze = planar.Length >= 0.2 ? planar.Normalized() : (Float2?)null;
            double t = Time.realtimeSinceStartupAsDouble;

            if (!_engine.TryGetOrigin(out double ox, out double oz))
            {
                ox = _engine.InitialOriginX(user);
                oz = _engine.InitialOriginZ(user);
            }
            depthSource.TryDispatch(config.map, config.depthStride, ox, oz, p.y, counts =>
                Submit(new Job { Counts = counts, User = user, Gaze = gaze, Time = t }));
        }

        void Submit(Job job)
        {
            if (!useWorkerThread)
            {
                Run(job);
                return;
            }
            Job dropped;
            lock (_mailboxLock)
            {
                dropped = _mailbox;
                _mailbox = job;
            }
            if (dropped != null) depthSource.Recycle(dropped.Counts);
            _wake.Set();
        }

        void WorkerLoop()
        {
            while (_running)
            {
                _wake.WaitOne();
                Job job;
                lock (_mailboxLock)
                {
                    job = _mailbox;
                    _mailbox = null;
                }
                if (job != null && _running) Run(job);
            }
        }

        void Run(Job job)
        {
            try
            {
                EngineResult result = _engine.ProcessCounts(job.Counts, job.User, job.Gaze, job.Time);
                Volatile.Write(ref _latest, result);
                Interlocked.Increment(ref _resultVersion);
            }
            catch (Exception e)
            {
                Debug.LogException(e);
            }
            finally
            {
                depthSource.Recycle(job.Counts);
            }
        }
    }
}
