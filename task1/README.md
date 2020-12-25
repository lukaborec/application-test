This solution is incomplete

This approach detects and removes text from the image and then extracts the shapes from it. The area of each shape is matched against the area of the bounding boxes of the detected tex; the text is rejected if the union of the text’s area and the shape’s area is not the same as the shape’s area. (in other words, the text is rejected if any part of its bounding box is outside of the shape)

Use as follows

```
python OCR.py —path=“path_to_image”
```

![solution](https://github.com/lukaborec/application-test/blob/main/task1/shapes.png)
![solution](https://github.com/lukaborec/application-test/blob/main/task1/solution.png)
