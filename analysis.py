import pandas as pd
from PIL import Image
import numpy as np
from math import prod

pixelMeans = pd.read_csv("pixelMeans.csv")

def createHeatmap(letter):
    img = Image.new( mode = "RGB", size = (128, 128) )
    pixels = img.load()
    for i in range(128):
        for j in range(128):
            val = round(pixelMeans[letter][128*j + i] * 255)
            color = (val, 0, val)
            pixels[i, j] = color

    img.save('test.png')

letters = pixelMeans.columns

def convert(img):
    imgArray = np.array(img)
    return np.array([1 if list(j) == [0, 0, 0] else 0 for i in imgArray for j in i])


def compare1(imgArray, means):
    score = 0
    for i in range(len(imgArray)):
        current = imgArray[i]
        currentMean = means[i]
        if current == 0:
            raw = 1 - currentMean
            if currentMean > 0:
                score -= raw
            else:
                score += raw
        else:
            raw = currentMean
            if currentMean == 0:
                score -= raw
            else:
                score += raw

    return score


def f(x, k, j): # x is 0 or 1, k is the k-1th pixel, starts at 0, j is a letter
    p = pixelMeans[j][k]
    scale = 10
    return scale*p if x == 1 else 1 - scale*p

def fj(x, j): # x is now an array, j is a letter
    g = float(prod([f(x[k], k, j) for k in range(128*128)]))
    return g

def P(j, x0):
    pi = 1/26
    total = sum([fj(x0, letter) for letter in letters])
    return fj(x0, j)/total

def arrayToImage(array, dim):
    imgbinary = array
    imgarray = np.array([[(0,0,0) for i in range(128)] for j in range(128)])
    for i in range(dim):
        for j in range(dim):
            if imgbinary[dim*i + j] == 0:
                imgarray[i][j] = (255, 255, 255)

    imgarray = imgarray.astype(np.uint8)
    return Image.fromarray(imgarray)


def nonbinaryArrayToImage(array, dim):
    imgarray = np.array([[(0,0,0) for i in range(dim)] for j in range(dim)])
    for i in range(dim):
        for j in range(dim):
            #if array[dim*i + j] > 0:
            #    print('bruh')
            imgarray[i][j] = (255*array[dim*i + j], 0, 0)

    imgarray = imgarray.astype(np.uint8)
    return Image.fromarray(imgarray)


def reduce(array, size, dim):
    print(array)
    reducedArray = np.array([i for i in range(dim*dim)])
    for i in range(0, 128, size):
        for j in range(0, 128, size):
            index = 128 * i + j
            #print(i, j)
            #print(index)
            #print(index, index+1, index+128, index+128+1)
            print([array[index], array[index +1], array[index + 128], array[index + 128 + 1]])
            bruh = np.mean(
                [array[index], array[index +1], array[index + 128], array[index + 128 + 1]]
            )
            #if bruh > 0:
            #    print(i, j, bruh)
            reducedArray[int(dim*i/size + j/size)] = bruh
            print(bruh)

    return reducedArray

img = Image.open(f"by_class/by_class/64/train_64/train_64_00029.png")
b = convert(img)
a = reduce(b, 2, 64)

print(np.where(a > 0))

nonbinaryArrayToImage(a, 32).save("reduceTest.png")

#nonbinaryArrayToImage(reduce(convert(Image.open(f"by_class/by_class/64/train_64/train_64_00029.png")), 4), 32).save("reduceTest.png")
    

def classify(img):
    array = convert(img)
    scores = [compare1(array, pixelMeans[letter]) for letter in letters]
    scoreDict = {letters[i]: scores[i] for i in range(26)}
    print(scoreDict)
    maxIndex = scores.index(max(scores))
    classifiedLetter = letters[maxIndex]
    return classifiedLetter

#if __name__ == "__main__":
    #img = Image.open(f"by_class/by_class/64/train_64/train_64_00029.png")
    #print(classify(img))
    
    #createHeatmap("d")