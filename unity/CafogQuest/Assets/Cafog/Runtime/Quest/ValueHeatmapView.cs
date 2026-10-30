using System.Threading;
using Cafog.Core;
using UnityEngine;

namespace Cafog.Quest
{
    /// <summary>
    /// Researcher/clinician view of the value field projected on the floor (off for patients).
    /// Shows the Parkinsonian value V_PD or the healthy critic V_HC as a TURBO-like colour map.
    /// </summary>
    public sealed class ValueHeatmapView : MonoBehaviour
    {
        [SerializeField] CafogEngineBehaviour engine;
        [SerializeField] Material material;
        [SerializeField] bool showHealthyCritic;
        [Range(0f, 1f)] [SerializeField] float opacity = 0.55f;
        [SerializeField] float refreshS = 0.2f;

        Texture2D _texture;
        Color32[] _pixels;
        float[] _values;
        Mesh _mesh;
        MeshRenderer _renderer;
        float _lastRefresh;
        double _originX, _originZ, _floorY;

        void Awake()
        {
            gameObject.AddComponent<MeshFilter>().sharedMesh = _mesh = new Mesh { name = "Cafog heatmap" };
            _renderer = gameObject.AddComponent<MeshRenderer>();
            _renderer.sharedMaterial = material;
            _renderer.enabled = false;
        }

        void Update()
        {
            if (engine == null || engine.Engine == null || Time.time - _lastRefresh < refreshS) return;
            _lastRefresh = Time.time;
            SpatialCueEngine e = engine.Engine;
            if (e.Field == null || !Monitor.TryEnter(e.FieldLock)) return;
            int n;
            double cs;
            try
            {
                n = e.Field.N;
                cs = e.Grid.CellSize;
                _originX = e.Grid.OriginX;
                _originZ = e.Grid.OriginZ;
                _floorY = e.Grid.FloorY ?? double.NaN;
                if (_values == null || _values.Length != n * n) _values = new float[n * n];
                System.Array.Copy(showHealthyCritic ? e.Field.ValueHc : e.Field.ValuePd, _values, n * n);
            }
            finally
            {
                Monitor.Exit(e.FieldLock);
            }
            Upload(n, (float)cs);
        }

        void Upload(int n, float cs)
        {
            if (_texture == null || _texture.width != n)
            {
                _texture = new Texture2D(n, n, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
                _pixels = new Color32[n * n];
                BuildQuad(n * cs);
                _renderer.material.mainTexture = _texture;
            }
            float lo = float.MaxValue, hi = float.MinValue;
            foreach (float v in _values)
            {
                if (v <= 0f) continue;
                lo = Mathf.Min(lo, v);
                hi = Mathf.Max(hi, v);
            }
            float span = Mathf.Max(hi - lo, 1e-6f);
            for (int i = 0; i < _values.Length; i++)
            {
                float v = _values[i];
                Color c = v > 0f ? Turbo((v - lo) / span) : Color.clear;
                c.a = v > 0f ? opacity : 0f;
                _pixels[i] = c;
            }
            _texture.SetPixels32(_pixels);
            _texture.Apply(false);
            float y = double.IsNaN(_floorY) ? 0f : (float)_floorY + 0.003f;
            transform.SetPositionAndRotation(new Vector3((float)_originX, y, (float)_originZ), Quaternion.identity);
            _renderer.enabled = true;
        }

        void BuildQuad(float size)
        {
            _mesh.Clear();
            _mesh.SetVertices(new[] { Vector3.zero, new Vector3(size, 0, 0), new Vector3(0, 0, size), new Vector3(size, 0, size) });
            _mesh.SetUVs(0, new[] { Vector2.zero, Vector2.right, Vector2.up, Vector2.one });
            _mesh.SetTriangles(new[] { 0, 2, 1, 1, 2, 3 }, 0);
            _mesh.RecalculateBounds();
        }

        // Polynomial approximation of Google's TURBO colormap.
        static Color Turbo(float x)
        {
            x = Mathf.Clamp01(x);
            float r = 0.13572138f + x * (4.61539260f + x * (-42.66032258f + x * (132.13108234f + x * (-152.94239396f + x * 59.28637943f))));
            float g = 0.09140261f + x * (2.19418839f + x * (4.84296658f + x * (-14.18503333f + x * (4.27729857f + x * 2.82956604f))));
            float b = 0.10667330f + x * (12.64194608f + x * (-60.58204836f + x * (110.36276771f + x * (-89.90310912f + x * 27.34824973f))));
            return new Color(Mathf.Clamp01(r), Mathf.Clamp01(g), Mathf.Clamp01(b));
        }
    }
}
