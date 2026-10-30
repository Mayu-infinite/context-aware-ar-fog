using System;
using System.Collections.Concurrent;
using Cafog.Core;
using UnityEngine;
using UnityEngine.Rendering;

namespace Cafog.Quest
{
    /// <summary>
    /// Bins the Meta Depth API frame into Cafog grid counts on the GPU and reads them back
    /// asynchronously. Requires an EnvironmentDepthManager (Meta XR Core SDK) in the scene,
    /// which publishes the depth texture and reprojection matrices as shader globals.
    /// </summary>
    public sealed class QuestDepthGridSource : MonoBehaviour
    {
        static readonly int DepthTextureId = Shader.PropertyToID("_EnvironmentDepthTexture");
        static readonly int ReprojectionId = Shader.PropertyToID("_EnvironmentDepthReprojectionMatrices");
        static readonly int ZBufferParamsId = Shader.PropertyToID("_EnvironmentDepthZBufferParams");

        [SerializeField] ComputeShader depthToGrid;
        [Tooltip("Flip the depth texture vertically if the grid appears mirrored front-to-back on device.")]
        [SerializeField] bool flipY;
        [SerializeField] float minRangeM = 0.1f;

        ComputeBuffer _counts;
        ComputeBuffer _floorState;
        int _kClear, _kHistogram, _kFloor, _kClassify;
        int _n;
        bool _pending;
        readonly ConcurrentStack<CellCounts> _pool = new ConcurrentStack<CellCounts>();

        public bool Busy => _pending;

        void Awake()
        {
            _kClear = depthToGrid.FindKernel("Clear");
            _kHistogram = depthToGrid.FindKernel("Histogram");
            _kFloor = depthToGrid.FindKernel("EstimateFloor");
            _kClassify = depthToGrid.FindKernel("Classify");
        }

        void OnDestroy()
        {
            _counts?.Release();
            _floorState?.Release();
        }

        /// <summary>Return a processed CellCounts for reuse (safe to call from any thread).</summary>
        public void Recycle(CellCounts counts)
        {
            if (counts != null && counts.N == _n) _pool.Push(counts);
        }

        /// <summary>
        /// Dispatches one depth frame binned against the given grid origin. onReady runs on the
        /// main thread when the GPU results arrive (typically 1-2 frames later).
        /// </summary>
        public bool TryDispatch(MapConfig map, int stride, double originX, double originZ, float eyeY, Action<CellCounts> onReady)
        {
            if (_pending) return false;
            var depth = Shader.GetGlobalTexture(DepthTextureId);
            if (depth == null) return false;
            Matrix4x4[] reprojection = Shader.GetGlobalMatrixArray(ReprojectionId);
            if (reprojection == null || reprojection.Length == 0) return false;

            EnsureBuffers(map.Cells);
            var cs = depthToGrid;
            int n = _n;
            cs.SetInt("_GridN", n);
            cs.SetFloat("_CellSize", (float)map.cellSizeM);
            cs.SetVector("_GridOrigin", new Vector4((float)originX, (float)originZ));
            cs.SetFloat("_FloorTolerance", (float)map.floorToleranceM);
            cs.SetFloat("_ObstacleMinHeight", (float)map.obstacleMinHeightM);
            cs.SetFloat("_ObstacleMaxHeight", (float)map.obstacleMaxHeightM);
            cs.SetFloat("_DropoffDepth", (float)map.dropoffDepthM);
            cs.SetFloat("_FloorSmoothing", (float)map.floorSmoothing);
            cs.SetFloat("_InitialEyeHeight", (float)map.initialEyeHeightM);
            cs.SetFloat("_MinRange", minRangeM);
            cs.SetFloat("_MaxRange", (float)map.maxRangeM);
            cs.SetFloat("_EyeY", eyeY);
            cs.SetInt("_Stride", Mathf.Max(1, stride));
            cs.SetInt("_FlipY", flipY ? 1 : 0);
            cs.SetInts("_DepthSize", depth.width, depth.height);
            cs.SetMatrix("_InvReprojection", reprojection[0].inverse);
            cs.SetVector("_DepthZBufferParams", Shader.GetGlobalVector(ZBufferParamsId));

            foreach (int k in new[] { _kClear, _kHistogram, _kFloor, _kClassify })
            {
                cs.SetBuffer(k, "_Counts", _counts);
                cs.SetBuffer(k, "_FloorState", _floorState);
            }
            cs.SetTexture(_kHistogram, "_DepthTex", depth);
            cs.SetTexture(_kClassify, "_DepthTex", depth);

            int total = 3 * n * n + FloorEstimator.Bins + 1;
            int gx = Mathf.CeilToInt(depth.width / (float)stride / 8f);
            int gy = Mathf.CeilToInt(depth.height / (float)stride / 8f);
            cs.Dispatch(_kClear, Mathf.CeilToInt(total / 64f), 1, 1);
            cs.Dispatch(_kHistogram, gx, gy, 1);
            cs.Dispatch(_kFloor, 1, 1, 1);
            cs.Dispatch(_kClassify, gx, gy, 1);

            _pending = true;
            AsyncGPUReadback.Request(_counts, request =>
            {
                _pending = false;
                if (request.hasError) return;
                var data = request.GetData<uint>();
                if (!_pool.TryPop(out CellCounts counts) || counts.N != n) counts = new CellCounts(n);
                int size = n * n;
                for (int i = 0; i < size; i++)
                {
                    counts.Obstacle[i] = (int)data[i];
                    counts.Floor[i] = (int)data[size + i];
                    counts.Dropoff[i] = (int)data[2 * size + i];
                }
                counts.OriginX = originX;
                counts.OriginZ = originZ;
                counts.FloorY = BitConverter.ToSingle(BitConverter.GetBytes(data[3 * size + FloorEstimator.Bins]), 0);
                onReady(counts);
            });
            return true;
        }

        void EnsureBuffers(int n)
        {
            if (_counts != null && _n == n) return;
            _counts?.Release();
            _floorState?.Release();
            _n = n;
            _pool.Clear();
            _counts = new ComputeBuffer(3 * n * n + FloorEstimator.Bins + 1, sizeof(uint));
            _floorState = new ComputeBuffer(2, sizeof(float));
            _floorState.SetData(new float[2]);
        }
    }
}
