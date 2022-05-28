import tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import h5py
import json

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=1)[0])

model.save('resnet50.h5')

extractor = tensorflow.keras.Model(
    inputs=model.inputs, outputs=[layer.output for layer in model.layers])
features = extractor(x)

f = h5py.File('feature.h5', 'w')
for i, layer in enumerate(model.layers):
    f[layer.name] = features[i][0]
f.close()

j = json.loads(model.to_json())
with open('model.json', 'w') as fp:
    json.dump(j, fp, indent=2)
