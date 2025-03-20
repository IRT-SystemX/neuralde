import numpy

from neural_de.transformations.centered_zoom import CenteredZoom
import numpy as np


class TestCenteredZoom:
    def test_transform_3_channels_size(self):
        keep_ratio = 0.6
        transformer = CenteredZoom(keep_ratio=keep_ratio)
        source_image = np.ones((10, 10, 3))
        target_image = np.ones((6, 6, 3))
        transformed_image = transformer.transform([source_image])[0]
        print("Source Image size", source_image.shape)
        print("Target Image size", target_image.shape)
        print("Transform Image size", transformed_image.shape)
        assert target_image.shape == transformed_image.shape

    def test_transform_3_channels_values(self):
        # IMAGE OF SIZE 10x10, KEEP_RATIO AT 0.6
        keep_ratio = 0.6
        transformer = CenteredZoom(keep_ratio=keep_ratio)
        FULL_SIZE = 10
        ZOOM_SIZE = int(keep_ratio * FULL_SIZE)
        # SOURCE IMAGE WITH PIXEL VALUE FOLLOWING: V(X,Y) = MAX(X,Y)
        source_image = np.ones((FULL_SIZE, FULL_SIZE, 3))
        for i in range(FULL_SIZE):
            for j in range(FULL_SIZE):
                value = max(i, j)
                source_image[i][j] = [value, value, value]
        source_image = np.array(source_image)
        # TARGET IMAGE WITH PIXEL VALUE FOLLOWING: V(X,Y) = MAX(X,Y) + (FULL_SIZE - ZOOM_SIZE)/2
        target_image = np.ones((ZOOM_SIZE, ZOOM_SIZE, 3))
        for k in range(ZOOM_SIZE):
            for l in range(ZOOM_SIZE):
                value = int(max(k, l) + (FULL_SIZE - ZOOM_SIZE) / 2)
                target_image[k][l] = [value, value, value]
        target_image = np.array(target_image)
        transformed_image = transformer.transform([source_image])[0]
        assert numpy.max(numpy.abs(transformed_image - target_image)) < 0.01

    def test_transform_with_annotations(self):
        keep_ratio = 0.8
        #Creation of the batch of images
        image1 = np.ones((100,100,3))
        image2 = np.ones((200,100,3))
        batch_images = [image1, image2]
        #Creation of the batch of bounding boxes
        bbox1 = [[20,20,50,40], [30,40,60,50]]
        bbox2 = [[100,50,150,40], [30,40,60,50]]
        batch_bbox = [bbox1, bbox2]
        #Target bbox
        target_bbox1 = [[10,10,40,30], [20,30,50,40]]
        target_bbox2 = [[80,40,130,30], [10,30,40,40]]
        target_batch = [target_bbox1, target_bbox2]
        #Test values
        transformer = CenteredZoom(keep_ratio=keep_ratio)
        results = transformer.transform_with_annotations(batch_images, batch_bbox)[1]
        assert np.max(np.abs(np.array(results) - np.array(target_batch))) < 0.01