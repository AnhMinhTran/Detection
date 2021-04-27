import imutils


# Slide windows accorss image
def slidind_win(img, step, ws):
    for i in range(0, img.shape[0] - ws[1], step):
        for x in range(0, img.shape[1] - ws[0], step):
            yield i, x, img[i:i + ws[1], x:x + ws[0]]


def img_pyramid(img, scale=1.5, minSize=(224, 224)):
    yield img

    while True:
        w = int(img.shape[1] / scale)
        img = imutils.resize(img, width=w)

        if img.shape[0] < minSize[1] or img.shape[1] < minSize[0]:
            break

        yield img
