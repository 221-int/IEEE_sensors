"""Camera-free checks: dashboard states and lost-input identity handling.

Run: python -m unittest src.v2.deploy.test_pi_demo_ui
"""
import copy
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.v2.deploy import pi_demo as demo


class RecordingPainter(demo.Painter):
    def __init__(self):
        super().__init__()
        self.labels = []

    def text(self, x, y, s, *args, **kwargs):
        self.labels.append(str(s))
        return super().text(x, y, s, *args, **kwargs)


class DashboardTests(unittest.TestCase):
    def test_display_preserves_color_and_projects_landmarks_without_mutating_input(self):
        frame = np.full((120, 160, 3), (30, 90, 180), np.uint8)
        original = frame.copy()
        mesh = np.full((468, 2), (80, 60), np.float32)
        meta = types.SimpleNamespace(center_x=80., center_y=60.,
                                     crop_w_px=100., crop_h_px=40., tilt_deg=25.)
        crop = demo.color_eye_crop(frame, mesh, meta)
        np.testing.assert_array_equal(crop[20, 20], (30, 90, 180))
        self.assertEqual(crop.shape, (96, 320, 3))
        np.testing.assert_array_equal(crop[48, 160], demo.GREEN)
        np.testing.assert_array_equal(frame, original)
        self.assertIsNone(demo.color_eye_crop(frame, mesh, None))

    def test_eye_crop_changes_only_the_rectangle(self):
        empty = demo.compose_eye_view()
        crop = np.full((64, 160), 83, np.uint8)
        actual = demo.compose_eye_view(crop)
        outside = np.ones(empty.shape[:2], dtype=bool)
        outside[220:316, 200:520] = False
        np.testing.assert_array_equal(actual[outside], empty[outside])
        np.testing.assert_array_equal(actual[226:310, 206:514], 83)

    def test_performance_footer_has_fps_and_actual_device(self):
        ui = RecordingPainter()
        state = demo.preview_state()
        state['device'] = 'Raspberry Pi 5 Model B Rev 1.1 · Python 3.11 · onnxruntime 1.19 · CPU only'
        demo.render(np.empty((720, 1280, 3), np.uint8), ui, state)
        self.assertIn('처리 14.7 fps', ui.labels)
        self.assertIn('Raspberry Pi 5 Model B Rev 1.1', ui.labels)
        for label in ui.labels:
            self.assertFalse(any(word in label for word in ('파이프라인', 'p50', 'p99', '33.3', 'CPU')))

    def test_missing_input_hides_stale_identity_and_live_metrics(self):
        for change in ({'face': False}, {'view': None},
                       {'stream': {'connected': False, 'err': 'Disconnected'}}):
            state = demo.preview_state()
            state.update(change)
            ui = RecordingPainter()
            demo.render(np.empty((720, 1280, 3), np.uint8), ui, state)
            self.assertIn('식별 대기', ui.labels)
            self.assertNotIn('328회', ui.labels)
            self.assertNotIn('2.4', ui.labels)

    def test_all_six_users_and_initial_baseline_visible(self):
        state = demo.preview_state()
        state['users'] = [(f'user{i}', f'사용자 {i}', .9) for i in range(1, 7)]
        state['baseline_personal'] = False
        ui = RecordingPainter()
        demo.render(np.empty((720, 1280, 3), np.uint8), ui, state)
        for i in range(1, 7):
            self.assertIn(f'사용자 {i}', ui.labels)
        self.assertIn('공통 초기 기준선', ui.labels)

    def test_canvas_sizes_and_preview_no_camera_or_profiles(self):
        state = demo.preview_state()
        for w, h in ((800, 480), (1280, 720), (1920, 1080), (1024, 768)):
            out = demo.render(np.empty((h, w, 3), np.uint8), demo.Painter(), state)
            self.assertEqual(out.shape, (h, w, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'preview.png')
            with patch.object(sys, 'argv', ['pi_demo', '--preview', path]), \
                    patch.object(demo, 'Profiles') as profiles, \
                    patch.object(demo, 'StreamReader') as stream:
                self.assertEqual(demo.main(), 0)
                profiles.assert_not_called()
                stream.assert_not_called()
            self.assertGreater(os.path.getsize(path), 1000)

    def test_lost_face_and_stream_clear_identity_before_history_update(self):
        frame = np.zeros((64, 80, 3), np.uint8)
        mesh = np.zeros((468, 2), np.float32)
        reader = MagicMock(connected=True, err='')
        reader.latest.side_effect = [(frame.copy(), i) for i in range(15)] + [(None, 15), (frame.copy(), 16)]
        frontend = MagicMock()
        frontend.mesh.side_effect = [mesh] * 14 + [None, mesh]
        frontend.crop_from_mesh.return_value = (np.zeros((64, 160), np.uint8),
            types.SimpleNamespace(center_x=40., center_y=32., crop_w_px=60., crop_h_px=24., tilt_deg=0.))
        enc, head = MagicMock(), MagicMock()
        enc.run.return_value = [np.ones((1, 16), np.float32)]
        head.run.return_value = [np.array([.05], np.float32)]
        ort = types.SimpleNamespace(SessionOptions=MagicMock,
                                    InferenceSession=MagicMock(side_effect=[enc, head]))
        fe_module = types.SimpleNamespace(EyeFrontend=MagicMock(return_value=frontend))
        states = []

        def capture(canvas, ui, state):
            states.append(copy.deepcopy(state))
            if len(states) == 17:
                raise KeyboardInterrupt
            return canvas

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'profiles.json')
            profiles = demo.Profiles(path)
            profiles.enroll('user1', '사용자 A', np.ones(16, np.float32))
            with patch.dict(sys.modules, {'onnxruntime': ort, 'src.v2.deploy.frontend': fe_module}), \
                    patch.object(sys, 'argv', ['pi_demo', '--headless', '--profiles', path, '--id-window', '10', '--blink-th', '0.5']), \
                    patch.object(demo, 'StreamReader', return_value=reader), \
                    patch.object(demo, 'render', side_effect=capture), \
                    patch.object(demo, 'read_temp', return_value=None), \
                    patch.object(demo, 'read_throttled', return_value=None), \
                    patch.object(demo, 'device_line', return_value='Test CPU'):
                self.assertEqual(demo.main(), 0)
            self.assertEqual(states[13]['ident_slot'], 'user1')
            for state in states[14:]:
                self.assertIsNone(state['ident_slot'])
                self.assertEqual(state['u_obs'], 0)
            np.testing.assert_array_equal(states[13]['view'], demo.compose_eye_view(np.zeros((64, 160), np.uint8)))
            np.testing.assert_array_equal(states[14]['view'], demo.compose_eye_view())
            self.assertIsNone(states[15]['view'])
            saved = demo.Profiles(path)
            self.assertAlmostEqual(saved.users['user1']['observed_sec'], states[13]['u_obs'])


if __name__ == '__main__':
    unittest.main()
