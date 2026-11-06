import unittest

import numpy as np

import extract_guidance_inputs as E


class GuidanceInputsTest(unittest.TestCase):
    def test_frame_round_trip(self):
        origin = (100.0, -50.0, 0.7)
        pts = np.array([[3.0, 1.0], [-2.0, 4.0]])
        self.assertTrue(np.allclose(E.global_to_ego(E.ego_to_global(pts, origin), origin), pts))

    def test_sdf_signs_and_scale_for_a_rotated_corridor(self):
        origin = (10.0, 20.0, 0.5)                                   # ego heading 0.5 rad in the global frame

        def inside_global(p):                                        # corridor |y_ego| < 4 in the ego frame
            return np.abs(E.global_to_ego(p, origin)[:, 1]) < 4.0

        sdf, sdf_origin, res = E.build_sdf(inside_global, origin)
        sdf = sdf.astype(np.float32)
        xs0, ys0 = sdf_origin
        row = lambda y: int(round((y - ys0) / res))
        col = lambda x: int(round((x - xs0) / res))
        self.assertAlmostEqual(float(sdf[row(0.0), col(10.0)]), 4.0, delta=0.3)      # centre: 4 m from either edge
        self.assertGreater(float(sdf[row(3.0), col(10.0)]), 0.0)
        self.assertLess(float(sdf[row(8.0), col(10.0)]), 0.0)
        self.assertAlmostEqual(float(sdf[row(8.0), col(10.0)]), -4.0, delta=0.3)

    def test_agents_are_rotated_into_the_ego_frame(self):
        origin = (0.0, 0.0, np.pi / 2)                               # ego looks along global +y
        agents = {"pos": np.array([[0.0, 10.0]]), "vel": np.array([[0.0, 3.0]]), "heading": np.array([np.pi / 2]),
                  "length": np.array([4.0]), "width": np.array([2.0])}
        e = E.agents_in_ego_frame(agents, origin)
        self.assertTrue(np.allclose(e["pos"], [[10.0, 0.0]], atol=1e-9))
        self.assertTrue(np.allclose(e["vel"], [[3.0, 0.0]], atol=1e-9))
        self.assertAlmostEqual(float(e["heading"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
