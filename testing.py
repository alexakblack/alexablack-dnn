import argparse

ap = argparse.ArgumentParser()

ap.add_argument(
	"-l",
	"--load-path",
	required=True,
	help="location to load model from",
	default=None,
)

ap.add_argument(
	"-p",
	"--path",
	required=False,
	help="path to image",
	default=None,
)

args = vars(ap.parse_args())
print(f"\n{args=}\n")

# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
import tensorflow as tf
import tensorflow.keras.models as models

classes = ["ADONIS", "AFRICAN GIANT SWALLOW", "AMERICAN SNOOT", "AN 88", "APOLLO", "ATALA", "BANDED ORANGE HELICONIAN", "BANDED PEACOCK", "BECKERS WHITE", "BLACK HAIRSTREAK", "BLUE MORPHO", "BLUE SPOTTED CROW", "BROWN SIPROETA", "CABBAGE WHITE", "CAIRNS BIRDWING", "CHECQUERED SKIPPER", "CHESTNUT", "CLEOPATRA"]
model = models.load_model(args["load_path"])

print(f"\n\nTensorflow version: {tf.__version__}")
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
import cv2
import numpy as np


def prep_image(image):
	image = cv2.resize(image, (224, 224))
	image = np.expand_dims(image, axis=0)
	return image


image = cv2.imread(args["path"])
if image is None:
    ap.error("invalid path to image")

image = prep_image(image)

prediction = list(model.predict(image)[0])
predictions_with_labels = list(zip(prediction, classes))
predictions_with_labels.sort(reverse=True, key=lambda x: x[0])

print("\n\nResults:")
for confidence, label in predictions_with_labels:
    print(f"{label : >8}: {confidence * 100:.2f}%")

cv2.destroyAllWindows()
