from sklearn.cluster import KMeans
import math
import numpy as np

def centroid_dict(x_train, typeList):
    c_dict = {}
    for key in typeList:
        x = np.copy(x_train[key])
        x = x.reshape((x.shape[0], -1))
        c_dict[key] = np.mean(x, axis=0)
    return c_dict

def getType(centroid, typeList, img_dict):
    closest_dist = -1
    closest_type = ""

    for key in typeList:
        image = img_dict[key]
        flatImg = image.flatten()
        dist = 0
        for i in range(len(flatImg)):
            dist += (flatImg[i] - centroid[i])**2
        dist = math.sqrt(dist)
        if (closest_dist < 0 or dist < closest_dist):
            closest_dist = dist
            closest_type = key
    return closest_type

def find_model(X, random_state, start, types, img_dict):
    kmean_model = KMeans (n_clusters=start, random_state = random_state)
    for i in range(start,(start+10)):
        kmean_model = KMeans (n_clusters=i, random_state = random_state)
        kmean_model.fit(X)
        #kmean_result =  kmean_model.predict(X)
        centroids = kmean_model.cluster_centers_
        typeList =[]
        for c in centroids:
            t = getType(c, types, img_dict)
            if t not in typeList:
                typeList.append(t)
            
        print(typeList, i)
        if len(typeList) == 6:
            break

    return kmean_model


