using System;

namespace Cafog.Core
{
    /// <summary>Ground-plane vector (world x, world z) in double precision.</summary>
    [Serializable]
    public struct Float2
    {
        public double X;
        public double Z;

        public Float2(double x, double z)
        {
            X = x;
            Z = z;
        }

        public static Float2 operator +(Float2 a, Float2 b) => new Float2(a.X + b.X, a.Z + b.Z);
        public static Float2 operator -(Float2 a, Float2 b) => new Float2(a.X - b.X, a.Z - b.Z);
        public static Float2 operator *(Float2 a, double s) => new Float2(a.X * s, a.Z * s);
        public static Float2 operator *(double s, Float2 a) => new Float2(a.X * s, a.Z * s);

        public double Dot(Float2 o) => X * o.X + Z * o.Z;
        public double Cross(Float2 o) => X * o.Z - Z * o.X;
        public double Length => Math.Sqrt(X * X + Z * Z);

        public Float2 Normalized()
        {
            double l = Length;
            return l > 1e-9 ? new Float2(X / l, Z / l) : this;
        }

        /// <summary>+1 for Unity's left-handed y-up world; -1 for a right-handed one (Python reference).</summary>
        public static double Handedness = 1.0;

        /// <summary>Right-hand side of a walking direction on the ground plane.</summary>
        public Float2 RightOf() => new Float2(Handedness * Z, -Handedness * X);

        public override string ToString() => $"({X:F2}, {Z:F2})";
    }
}
