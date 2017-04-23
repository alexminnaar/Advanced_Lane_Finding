from lane_finding_pipeline import LaneFindingPipeline


class Line:
    def __init__(self):
        self.pipeline = LaneFindingPipeline()
        self.past_left_lane = []
        self.past_right_lane = []
        self.past_left_curvature = []
        self.past_right_curvature = []

    def process_image(self, img):
        lane_img, left_x, right_x, left_curve, right_curve = self.pipeline.apply_pipeline(img)


