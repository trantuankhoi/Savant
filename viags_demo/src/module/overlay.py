import cv2
import yaml
from time import perf_counter as time
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import BBox, NvDsFrameMeta
from savant.utils.artist import Artist, Position
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with open(self.config_path, 'r', encoding='utf8') as stream:
            self.area_config = yaml.safe_load(stream)

        self.areas = {}
        for source_id, areas in self.area_config.items():
            self.areas[source_id] = {}
            for area_name, area_dict in areas.items():
                self.areas[source_id][area_name] = (
                    area_dict['points'],
                    area_dict['color'],
                )

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        # manually refresh (by filling with black) frame padding used for drawing
        # this workaround avoids rendering problem where drawings from previous frames
        # are persisted on the padding area in the next frame
        # frame_w, _ = artist.frame_wh
        if frame_meta.source_id not in self.areas:
            return

        primary_meta_object = None
        # obj_metas = []
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                primary_meta_object = obj_meta
            elif obj_meta.label == self.target_obj_label:
                # t0 = time()
                artist.add_bbox(obj_meta.bbox, 3, (0, 255, 0, 255))
                # logger.info(f"Add bbox: {time() - t0}")
                # obj_metas.append(obj_meta)
                # self.visualize(obj_meta.bbox)

        if not primary_meta_object:
            return

        # legend_rect_left = frame_w - self.sidebar_width + 75
        # legend_rect_width = 50
        # legend_rect_center_x = legend_rect_left + legend_rect_width // 2
        # legend_text_x = legend_rect_left + legend_rect_width + 20
        # legend_y = 50

        # area_lines = self.areas[frame_meta.source_id]
        # logger.info(f"KhoiTT: {area_lines}")
        # for area_name, (points, color) in area_lines.items():
            # t0 = time()
            # artist.add_polygon(points, line_color=color, line_width=2)
            # logger.info(f"Add polygon: {time() - t0}")
            # for obj in obj_metas:
            #     if obj.draw_label == area_name:
            #         center = round(obj.bbox.xc), round(obj.bbox.yc)
            #         top_center = round(obj.bbox.xc), round(obj.bbox.yc - obj.bbox.height / 2)
            #         artist.add_circle(center, 3, color, cv2.FILLED)
            #         artist.add_text("#" + str(obj.track_id), top_center, anchor_point_type=Position.CENTER_TOP)

            # n_objs_meta = primary_meta_object.get_attr_meta('analytics', area_name)
            # if n_objs_meta:
            #     n_objs = n_objs_meta.value
            # else:
            #     n_objs = 0

            # artist.add_text(
            #     f'{n_objs:2d}',
            #     (legend_text_x, legend_y),
            #     2,
            #     4,
            #     anchor_point_type=Position.CENTER,
            # )

            # artist.add_bbox(
            #     BBox(
            #         legend_rect_center_x,
            #         legend_y + text_size[1] // 2,
            #         legend_rect_width,
            #         legend_rect_width,
            #     ),
            #     border_width=0,
            #     bg_color=color,
            # )

            # legend_y += legend_rect_width + 50

    
    def visualize(self, results):
        image = cv2.imread("/test_data/test_img.jpeg")
        input_w , input_h = 640, 640
        img_h, img_w, _ = image.shape
        r_w = input_w / img_w
        r_h = input_h / img_h
        r = min(r_w, r_h)

        # for box in results:
        xc = int(results.xc)
        yc = int(results.yc)
        w = int(results.width)
        h = int(results.height)
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        # y1 = int(yc - h / 2 - (input_h - r * img_h) / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)
        # y2 = int(yc + h / 2 - (input_h - r * img_h) / 2)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite("/opt/savant/src/test_img_result.jpg", image)