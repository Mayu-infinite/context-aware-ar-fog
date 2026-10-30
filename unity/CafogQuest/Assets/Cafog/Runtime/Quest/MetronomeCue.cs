using UnityEngine;

namespace Cafog.Quest
{
    /// <summary>
    /// Auditory cue: a click at the LIP step rhythm (paper Eq. 25) while the visual cue is active.
    /// Beats are scheduled on the DSP clock so they stay sample-accurate under frame drops.
    /// </summary>
    public sealed class MetronomeCue : MonoBehaviour
    {
        [SerializeField] CafogEngineBehaviour engine;
        [SerializeField] AudioClip click;
        [Range(0f, 1f)] [SerializeField] float volume = 0.6f;
        [SerializeField] double scheduleAheadS = 0.1;

        AudioSource[] _sources;
        int _next;
        double _nextBeat;
        bool _playing;

        void Awake()
        {
            if (click == null) click = MakeClick();
            _sources = new AudioSource[2];
            for (int i = 0; i < _sources.Length; i++)
            {
                _sources[i] = gameObject.AddComponent<AudioSource>();
                _sources[i].clip = click;
                _sources[i].playOnAwake = false;
                _sources[i].spatialBlend = 0f;
                _sources[i].volume = volume;
            }
        }

        void Update()
        {
            var result = engine != null ? engine.Latest : null;
            double bpm = result != null && result.Cue.Active ? result.Cue.MetronomeBpm : 0.0;
            double now = AudioSettings.dspTime;
            if (bpm <= 0)
            {
                _playing = false;
                return;
            }
            if (!_playing)
            {
                _playing = true;
                _nextBeat = now + 0.05;
            }
            while (_nextBeat < now + scheduleAheadS)
            {
                _sources[_next].PlayScheduled(_nextBeat);
                _next = (_next + 1) % _sources.Length;
                _nextBeat += 60.0 / bpm;
            }
        }

        static AudioClip MakeClick()
        {
            const int rate = 48000;
            int n = rate / 40;
            var data = new float[n];
            for (int i = 0; i < n; i++)
            {
                float t = i / (float)rate;
                data[i] = Mathf.Sin(2f * Mathf.PI * 1200f * t) * Mathf.Exp(-t * 180f);
            }
            var clip = AudioClip.Create("Cafog click", n, 1, rate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
