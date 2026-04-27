# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import logging
import os
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import mujoco
import trimesh
from urdfpy import URDF
from dm_control.utils.transformations import quat_to_euler

from .renderer import Renderer

logger = logging.getLogger(__name__)


def _get_default_config(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)


def get_digit_config_path():
    return _get_default_config("config_digit.yml")


def get_digit_shadow_config_path():
    return _get_default_config("config_digit_shadow.yml")


def get_omnitact_config_path():
    return _get_default_config("config_omnitact.yml")


@dataclass
class Link:
    mj_data: mujoco.MjData
    mj_model: mujoco.MjModel
    obj_id: int  # mujoco ID
    link_id: int  # mujoco link ID (-1 means base)
    cid: int  # physicsClientId
    ordering: str

    def get_pose(self):

        geom_id = np.where(self.mj_model.geom_bodyid == self.obj_id)[0][0]


        position = self.mj_data.geom_xpos[geom_id].copy()
        # orientation = self.mj_data.geom_xmat[geom_id].copy()

        position = self.mj_data.xpos[self.obj_id]
        # orientation = self.mj_data.xmat[self.obj_id]
        orientation = self.mj_data.xquat[self.obj_id]

        # orientation[0], orientation[1], orientation[2], orientation[3] = orientation[1], orientation[2], orientation[3], orientation[0]
        # orientation = rmat_to_euler(orientation.reshape(3, 3), ordering="YXZ")
        orientation = quat_to_euler(orientation, ordering=self.ordering)

        return position, orientation

class Sensor:
    def __init__(
            self,
            mj_model: mujoco.MjModel,
            mj_data: mujoco.MjData,
            width=120,
            height=160,
            background=None,
            config_path=get_digit_config_path(),
            visualize_gui=True,
            show_depth=True,
            zrange=0.002,
            cid=0,
            ordering="XYZ"
    ):
        """

        :param width: scalar
        :param height: scalar
        :param background: image
        :param visualize_gui: Bool
        :param show_depth: Bool
        :param config_path:
        :param cid: Int
        """
        self.cid = cid
        self.renderer = Renderer(width, height, background, config_path)

        self.mj_model = mj_model
        self.mj_data = mj_data

        self.visualize_gui = visualize_gui
        self.show_depth = show_depth
        self.zrange = zrange

        self.cameras = {}
        self.nb_cam = 0
        self.objects = {}
        self.object_poses = {}
        self.normal_forces = {}
        self._static = None

        self.ordering = ordering

    @property
    def height(self):
        return self.renderer.height

    @property
    def width(self):
        return self.renderer.width

    @property
    def background(self):
        return self.renderer.background

    def add_camera(self, obj_id, link_ids):
        """
        Add camera into tacto

        self.cameras format: {
            "cam0": Link,
            "cam1": Link,
            ...
        }
        """
        if not isinstance(link_ids, collections.abc.Sequence):
            link_ids = [link_ids]

        for link_id in link_ids:
            cam_name = "cam" + str(self.nb_cam)
            self.cameras[cam_name] = Link(self.mj_data, self.mj_model, obj_id, link_id, self.cid, self.ordering)
            self.nb_cam += 1

    def add_object(self, urdf_fn, obj_id, mujococ_link_id, globalScaling=1.0):
        # Load urdf file by urdfpy
        robot = URDF.load(urdf_fn)
        for link_id, link in enumerate(robot.links):
            if len(link.visuals) == 0:
                continue
            # Get each links
            visual = link.visuals[0]
            obj_trimesh = visual.geometry.meshes[0]

            # Set mesh color to default (remove texture)
            obj_trimesh.visual = trimesh.visual.ColorVisuals()

            # Set initial origin (pybullet pose already considered initial origin position, not orientation)
            pose = visual.origin

            # Scale if it is mesh object (e.g. STL, OBJ file)
            mesh = visual.geometry.mesh
            if mesh is not None and mesh.scale is not None:
                S = np.eye(4, dtype=np.float64)
                S[:3, :3] = np.diag(mesh.scale)
                pose = pose.dot(S)

            # Apply interial origin if applicable
            inertial = link.inertial

            if inertial is not None and inertial.origin is not None:
                # TODO inertial multiplication ??

                # inertial.origin = np.matmul(inertial.origin, np.linalg.inv(inertial.origin))

                # pose = np.linalg.inv(inertial.origin).dot(pose)
                pass

            # Set global scaling
            pose = np.diag([globalScaling] * 3 + [1]).dot(pose)

            obj_trimesh = obj_trimesh.apply_transform(pose)
            obj_name = "{}_{}".format(obj_id, mujococ_link_id)

            self.objects[obj_name] = Link(self.mj_data, self.mj_model, obj_id, mujococ_link_id, self.cid, self.ordering)
            position, orientation = self.objects[obj_name].get_pose()

            # Add object in pyrender
            self.renderer.add_object(
                obj_trimesh,
                obj_name,
                position=position,  # [-0.015, 0, 0.0235],
                orientation=orientation,  # [0, 0, 0],
            )

    def add_body(self, urdf_path, id, link_id):
        self.add_object(
            urdf_path, id, link_id, globalScaling=1.0
        )

    def update(self):
        warnings.warn(
            "\33[33mSensor.update is deprecated and renamed to ._update_object_poses()"
            ", which will be called automatically in .render()\33[0m"
        )

    def _update_object_poses(self):
        """
        Update the pose of each objects registered in tacto simulator
        """
        for obj_name in self.objects.keys():
            self.object_poses[obj_name] = self.objects[obj_name].get_pose()

    def get_force(self, cam_name):
        # Load contact force

        obj_id = self.cameras[cam_name].obj_id
        link_id = self.cameras[cam_name].link_id

        pts = self.mj_data.contact

        # accumulate forces from 0. using defaultdict of float
        self.normal_forces[cam_name] = collections.defaultdict(float)

        for index, pt in enumerate(pts):
            if self.mj_model.geom(pt.geom2).bodyid[0] != obj_id:
                if obj_id == self.mj_model.geom(pt.geom1).bodyid[0]:
                    interesting_geom = pt.geom2

                else:
                    continue
            else:
                interesting_geom = pt.geom1

            body_id_b = self.mj_model.geom(interesting_geom).bodyid[0]

            link_id_b = self.mj_model.body(self.mj_model.geom(interesting_geom).bodyid[0]).parentid[0]

            obj_name = "{}_{}".format(body_id_b, link_id_b)
            # ignore contacts we don't care (those not in self.objects)
            if obj_name not in self.objects:
                continue

            data = np.zeros(6)
            raw_model = getattr(self.mj_model, '_model', self.mj_model)
            raw_data = getattr(self.mj_data, '_data', self.mj_data)
            mujoco.mj_contactForce(raw_model, raw_data, index, data)
            # print(np.linalg.norm(data[:3]))
            # Accumulate normal forces
            self.normal_forces[cam_name][obj_name] += np.linalg.norm(data[:3])

        return self.normal_forces[cam_name]

    @property
    def static(self):
        if self._static is None:
            colors, _ = self.renderer.render(noise=False)
            depths = [np.zeros_like(d0) for d0 in self.renderer.depth0]
            self._static = (colors, depths)

        return self._static

    def _render_static(self):
        colors, depths = self.static
        colors = [self.renderer._add_noise(color) for color in colors]
        return colors, depths

    def render(self, visualize_digit: bool = False):
        """
        Render tacto images from each camera's view.
        """

        self._update_object_poses()

        colors = []
        depths = []

        for i in range(self.nb_cam):
            cam_name = "cam" + str(i)

            # get the contact normal forces
            normal_forces = self.get_force(cam_name)

            if normal_forces:
                position, orientation = self.cameras[cam_name].get_pose()
                self.renderer.update_camera_pose(position, orientation)
                color, depth = self.renderer.render(self.object_poses, normal_forces, visualize_scene=visualize_digit)

                # Remove the depth from curved gel
                for j in range(len(depth)):
                    depth[j] = self.renderer.depth0[j] - depth[j]
            else:
                color, depth = self._render_static()

            colors += color
            depths += depth

        return colors, depths

    def _depth_to_color(self, depth):
        gray = (np.clip(depth / self.zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def updateGUI(self, colors, depths):
        """
        Update images for visualization
        """
        if not self.visualize_gui:
            return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)

        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)

            # concatenate the resulting two images vertically (axis=0)
            color_n_depth = np.concatenate([color, depth], axis=0)

            cv2.imshow(
                "color and depth", cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR)
            )
        else:
            cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)
