"""Draws the rig with raylib.

Owns the GPU resources -- one unit cube reused for every bone, one material, and
a small lit shader. raylib's default shader does no lighting, which would render
the robot as flat silhouettes where no edge is readable, so this supplies a
directional light plus a rim term that separates the puppet from the backdrop.

Bone transforms are composed on the CPU and handed to ``draw_mesh`` as a 4x4.
About two dozen matrices per frame is free, and it avoids needing the rlgl
matrix stack, which this binding does not expose.
"""

from __future__ import annotations

import math

import pyray as rl

from gesture_ai.rig import SKELETON, Bone, JointPose

_VERTEX_SHADER = """
#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
uniform mat4 mvp;
uniform mat4 matModel;
uniform mat4 matNormal;
out vec3 fragNormal;
out vec3 fragPos;
void main()
{
    fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
    fragNormal = normalize(vec3(matNormal * vec4(vertexNormal, 1.0)));
    gl_Position = mvp * vec4(vertexPosition, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
in vec3 fragNormal;
in vec3 fragPos;
uniform vec4 colDiffuse;
uniform vec3 lightDir;
uniform vec3 viewPos;
uniform float emissive;
out vec4 finalColor;
void main()
{
    vec3 n = normalize(fragNormal);
    vec3 l = normalize(-lightDir);
    vec3 v = normalize(viewPos - fragPos);
    vec3 h = normalize(l + v);

    float diff = max(dot(n, l), 0.0);
    float spec = pow(max(dot(n, h), 0.0), 24.0) * 0.30;
    // Rim term: brightest where the surface turns away, which draws the
    // silhouette and keeps the puppet legible over a busy camera backdrop.
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.55;

    vec3 base = colDiffuse.rgb;
    vec3 color = base * (0.30 + 0.80 * diff)
               + vec3(1.0, 0.97, 0.92) * spec
               + vec3(0.35, 0.48, 0.95) * rim;
    color = mix(color, base * 1.7 + vec3(0.15), emissive);
    finalColor = vec4(color, colDiffuse.a);
}
"""

_LIGHT_DIR = (-0.45, -0.75, -0.5)


def _matrix_from_basis(basis) -> rl.Matrix:
    """Build a raylib rotation matrix from a ``(right, up, normal)`` frame.

    raylib composes with row vectors, so each basis vector -- the image of one
    axis -- occupies a row: (m0,m4,m8), (m1,m5,m9), (m2,m6,m10).
    """
    right, up, normal = basis
    m = rl.matrix_identity()
    m.m0, m.m4, m.m8 = right
    m.m1, m.m5, m.m9 = up
    m.m2, m.m6, m.m10 = normal
    return m


def _radians3(degrees) -> rl.Vector3:
    return rl.Vector3(
        math.radians(degrees[0]), math.radians(degrees[1]), math.radians(degrees[2])
    )


class PuppetRenderer:
    """Holds the mesh, material and shader; draws one posed skeleton per frame."""

    def __init__(self) -> None:
        self._mesh = rl.gen_mesh_cube(1.0, 1.0, 1.0)
        self._material = rl.load_material_default()
        self._shader = rl.load_shader_from_memory(_VERTEX_SHADER, _FRAGMENT_SHADER)
        self._material.shader = self._shader

        self._loc_light = rl.get_shader_location(self._shader, "lightDir")
        self._loc_view = rl.get_shader_location(self._shader, "viewPos")
        self._loc_emissive = rl.get_shader_location(self._shader, "emissive")

        rl.set_shader_value(
            self._shader,
            self._loc_light,
            rl.ffi.new("float[3]", list(_LIGHT_DIR)),
            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3,
        )
        self._world: dict[str, rl.Matrix] = {}

    def unload(self) -> None:
        rl.unload_shader(self._shader)
        rl.unload_mesh(self._mesh)

    def _set_emissive(self, value: float) -> None:
        rl.set_shader_value(
            self._shader,
            self._loc_emissive,
            rl.ffi.new("float*", value),
            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT,
        )

    def solve_world(self, joints: JointPose) -> dict[str, rl.Matrix]:
        """Resolve every bone to a world matrix.

        One forward pass: ``SKELETON`` is ordered parents-first, so a parent's
        matrix is always resolved before any child needs it.
        """
        world: dict[str, rl.Matrix] = {}
        for bone in SKELETON:
            rotation = joints.rotations.get(bone.name, (0.0, 0.0, 0.0))
            offset = bone.offset
            if bone.parent is None:
                offset = (
                    offset[0] + joints.root_translation[0],
                    offset[1] + joints.root_translation[1],
                    offset[2] + joints.root_translation[2],
                )
            local = rl.matrix_rotate_xyz(_radians3(rotation))
            basis = joints.matrices.get(bone.name)
            if basis is not None:
                # Matrix orientation first, then the euler term as a secondary
                # offset on top of it.
                local = rl.matrix_multiply(_matrix_from_basis(basis), local)
            local = rl.matrix_multiply(
                local, rl.matrix_translate(offset[0], offset[1], offset[2])
            )
            world[bone.name] = (
                local
                if bone.parent is None
                else rl.matrix_multiply(local, world[bone.parent])
            )
        self._world = world
        return world

    def draw(self, joints: JointPose, camera: rl.Camera3D) -> None:
        rl.set_shader_value(
            self._shader,
            self._loc_view,
            rl.ffi.new(
                "float[3]",
                [camera.position.x, camera.position.y, camera.position.z],
            ),
            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC3,
        )

        world = self.solve_world(joints)
        diffuse = self._material.maps[rl.MaterialMapIndex.MATERIAL_MAP_ALBEDO]

        for bone in SKELETON:
            if bone.size == (0, 0, 0):
                continue
            self._set_emissive(joints.glow if bone.glow else 0.0)
            r, g, b = bone.color
            diffuse.color = rl.Color(int(r), int(g), int(b), 255)
            transform = rl.matrix_multiply(
                rl.matrix_multiply(
                    rl.matrix_scale(bone.size[0], bone.size[1], bone.size[2]),
                    rl.matrix_translate(bone.center[0], bone.center[1], bone.center[2]),
                ),
                world[bone.name],
            )
            rl.draw_mesh(self._mesh, self._material, transform)

    def bone_origin(self, name: str) -> rl.Vector3:
        """World position of a bone's joint, for effects anchored to the rig."""
        m = self._world.get(name)
        if m is None:
            return rl.Vector3(0.0, 0.0, 0.0)
        return rl.Vector3(m.m12, m.m13, m.m14)


# Root height at which the feet touch the ground; above this the puppet is
# airborne and the shadow should shrink away from it.
_CONTACT_HEIGHT = 2.7


def ground_shadow(joints: JointPose) -> tuple[rl.Vector3, float]:
    """Position and radius for a fake contact shadow under the puppet."""
    x, y, z = joints.root_translation
    lift = max(0.0, y - _CONTACT_HEIGHT)
    # Shrinks as the puppet rises, which is what sells the height.
    radius = 2.3 / (1.0 + lift * 0.22)
    return rl.Vector3(x, 0.02, z), radius
