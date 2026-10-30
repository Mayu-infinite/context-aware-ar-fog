using System.Collections.Generic;
using Cafog.Core;
using UnityEngine;

namespace Cafog.Quest
{
    /// <summary>
    /// Draws the engine's footsteps as world-anchored footprints (or transverse bars) on the
    /// real floor. Nothing is labelled: the patient only sees where to step next.
    /// </summary>
    public sealed class FootstepCueRenderer : MonoBehaviour
    {
        public enum CueStyle { Footprints, TransverseBars }

        [SerializeField] CafogEngineBehaviour engine;
        [Tooltip("Unlit transparent material, e.g. Cafog/FootprintUnlit.")]
        [SerializeField] Material material;
        [SerializeField] CueStyle style = CueStyle.Footprints;
        [SerializeField] Color leftColor = new Color(0.25f, 0.95f, 1f, 0.9f);
        [SerializeField] Color rightColor = new Color(1f, 0.85f, 0.25f, 0.9f);
        [SerializeField] float footLengthM = 0.26f;
        [SerializeField] float barLengthM = 0.6f;
        [SerializeField] float fadeInS = 0.25f;
        [SerializeField] float fadeOutS = 0.6f;
        [Tooltip("Brightness of steps after the next one (the next step is always full strength).")]
        [Range(0f, 1f)] [SerializeField] float laterStepAlpha = 0.55f;
        [SerializeField] float floorOffsetM = 0.006f;

        static readonly int ColorId = Shader.PropertyToID("_Color");

        readonly List<MeshRenderer> _pool = new List<MeshRenderer>();
        MaterialPropertyBlock _block;
        Mesh _leftMesh, _rightMesh, _barMesh;
        List<Footstep> _shown = new List<Footstep>();
        float _visibility;
        float _target;
        double _floorY = double.NaN;
        int _version = -1;

        void Awake()
        {
            _block = new MaterialPropertyBlock();
            _rightMesh = FootprintMesh(footLengthM, mirror: false);
            _leftMesh = FootprintMesh(footLengthM, mirror: true);
            _barMesh = BarMesh(barLengthM, 0.06f);
        }

        void Update()
        {
            EngineResult result = engine != null ? engine.Latest : null;
            if (result != null && engine.ResultVersion != _version)
            {
                _version = engine.ResultVersion;
                if (!double.IsNaN(result.FloorY)) _floorY = result.FloorY;
                if (result.Cue.Footsteps.Count > 0) _shown = result.Cue.Footsteps;
                _target = result.Cue.Active && result.Cue.Footsteps.Count > 0 ? 1f : 0f;
            }
            float rate = _target > _visibility ? 1f / Mathf.Max(fadeInS, 1e-3f) : 1f / Mathf.Max(fadeOutS, 1e-3f);
            _visibility = Mathf.MoveTowards(_visibility, _target, rate * Time.deltaTime);
            if (_visibility <= 0f) _shown = new List<Footstep>();
            Draw();
        }

        void Draw()
        {
            float floorY = double.IsNaN(_floorY) ? 0f : (float)_floorY;
            while (_pool.Count < _shown.Count) _pool.Add(CreateStep());
            for (int i = 0; i < _pool.Count; i++)
            {
                MeshRenderer r = _pool[i];
                bool on = i < _shown.Count && _visibility > 0f;
                r.enabled = on;
                if (!on) continue;
                Footstep step = _shown[i];
                var forward = new Vector3((float)step.Forward.X, 0f, (float)step.Forward.Z);
                r.transform.SetPositionAndRotation(
                    new Vector3((float)step.Position.X, floorY + floorOffsetM, (float)step.Position.Z),
                    Quaternion.LookRotation(forward, Vector3.up));
                r.GetComponent<MeshFilter>().sharedMesh =
                    style == CueStyle.TransverseBars ? _barMesh : step.Side < 0 ? _leftMesh : _rightMesh;
                Color c = step.Side < 0 ? leftColor : rightColor;
                c.a *= _visibility * (i == 0 ? 1f : laterStepAlpha);
                _block.SetColor(ColorId, c);
                r.SetPropertyBlock(_block);
            }
        }

        MeshRenderer CreateStep()
        {
            var go = new GameObject("Cafog footstep");
            go.transform.SetParent(transform, false);
            go.AddComponent<MeshFilter>();
            var r = go.AddComponent<MeshRenderer>();
            r.sharedMaterial = material;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            return r;
        }

        // Footprint outline (right foot, +x lateral, +z forward), normalised to length 1.
        static readonly Vector2[] Outline =
        {
            new Vector2(0.00f, -0.50f), new Vector2(0.12f, -0.46f), new Vector2(0.17f, -0.35f),
            new Vector2(0.16f, -0.15f), new Vector2(0.15f, 0.02f), new Vector2(0.21f, 0.18f),
            new Vector2(0.22f, 0.33f), new Vector2(0.18f, 0.44f), new Vector2(0.08f, 0.50f),
            new Vector2(-0.05f, 0.48f), new Vector2(-0.15f, 0.38f), new Vector2(-0.18f, 0.22f),
            new Vector2(-0.14f, 0.05f), new Vector2(-0.10f, -0.12f), new Vector2(-0.13f, -0.32f),
            new Vector2(-0.11f, -0.45f),
        };

        // Fan around the centre; uv.x = 1 at the centre and 0 on the rim for a soft edge.
        static Mesh FootprintMesh(float length, bool mirror)
        {
            int n = Outline.Length;
            var vertices = new Vector3[n + 1];
            var uvs = new Vector2[n + 1];
            var triangles = new int[3 * n];
            vertices[0] = new Vector3(0f, 0f, 0.02f * length);
            uvs[0] = new Vector2(1f, 0f);
            for (int i = 0; i < n; i++)
            {
                Vector2 p = Outline[i];
                vertices[i + 1] = new Vector3((mirror ? -p.x : p.x) * length, 0f, p.y * length);
                uvs[i + 1] = Vector2.zero;
                int a = i + 1, b = (i + 1) % n + 1;
                triangles[3 * i] = 0;
                triangles[3 * i + 1] = mirror ? b : a;
                triangles[3 * i + 2] = mirror ? a : b;
            }
            var mesh = new Mesh { name = mirror ? "Cafog left foot" : "Cafog right foot" };
            mesh.SetVertices(vertices);
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(triangles, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        static Mesh BarMesh(float length, float depth)
        {
            float hx = 0.5f * length, hz = 0.5f * depth;
            var mesh = new Mesh { name = "Cafog transverse bar" };
            mesh.SetVertices(new[]
            {
                new Vector3(0f, 0f, 0f), new Vector3(-hx, 0f, -hz), new Vector3(hx, 0f, -hz),
                new Vector3(hx, 0f, hz), new Vector3(-hx, 0f, hz),
            });
            mesh.SetUVs(0, new[] { new Vector2(1f, 0f), Vector2.zero, Vector2.zero, Vector2.zero, Vector2.zero });
            mesh.SetTriangles(new[] { 0, 1, 4, 0, 4, 3, 0, 3, 2, 0, 2, 1 }, 0);
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
