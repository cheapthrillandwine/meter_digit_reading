# meter_digit_reading
This is my research. I will try to commit frequently. I appriciate you informing issue me for my study !
## AIM
Read digital meter reading using deeplearning
## ENVIRONMENT
- Windows7-64bit
- Anaconda
- Python 3.6
- OpenCV 3.4
- Numpy 1.14
- scikit-learn 0.19.1
- Tensorflow 1.6.0
- Keras 2.1.3

### MNIST APPLIED
仮説：メーターの数字は表現量が少ないため、単純なCNNに複数の文字フォントを用いたデータセットを学習させることで認識が可能になる。

学習フォント：Windows内蔵フォント+[DSEG](https://www.keshikan.net/fonts.html)
フォント数:467
数字パターン:380052
テストデータ:1412
モデル:MLP(Multi-Layer Perceptron)
バッチ:128
エポック:50
## STEP1: IMAGE PREPROCESSING

```python
# Gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
# Bilateral filter
blur = cv2.bilateralFilter(gray, 15, 20, 20) # bilateral Filter
cv2.imshow("blur", blur)
# Ostu-threshold
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
```
## STEP2: NEGATIVE POSITIVE INVERSION

```python
# Black pixels counting
num_b = np.count_nonzero(thresh)
print(num_b)

# White pixels counting
num_w = (thresh.size) - num_b
print(num_w)

if num_w < num_b:
    thresh = cv2.bitwise_not(thresh)
```

## STEP3: EXCEPT NON-RELATED

```python
rects = []
img_w = img.shape[1]
# Coordinate extract
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    # Size threshold setting
    if w < 5: continue
    if h < 5: continue
    if w > img_w / 5: continue

    # Range using y2 coordinate
    y2 = round(y / 10) * 10
    index = y2 * img_w + x
    rects.append((index, x, y, w, h))
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
# Sort using x coordinate
rects = sorted(rects, key=lambda x:x[0])
cv2.imshow("rectangle", img)
```
