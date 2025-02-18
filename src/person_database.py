import faiss
from src.person import Person
from models.feature_extractor.custom_feature_extractor import CustomFeatureExtractor
import cv2
import numpy as np


def preprocess_images(image_list, target_size=(224, 224)):
    """
    对输入的多张人物图像进行预处理，增强人物特征，减少背景噪声。

    Parameters
    ----------
    image_list: list
        输入的图像列表，每个图像是一个 OpenCV 格式的图像（BGR格式）。

    target_size: tuple
        图像缩放的目标大小，默认为 (224, 224)，适合输入深度学习模型。

    Returns
    -------
    image_arrays: list
        经过预处理后的图像 numpy.ndarray 列表，可以直接用于特征提取。
    """
    image_arrays = []

    for image in image_list:
        # 1. 去噪处理（高斯滤波）
        img_denoised = cv2.GaussianBlur(image, (5, 5), 0)

        # 2. 锐化处理（拉普拉斯算子）
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_sharpened = cv2.filter2D(img_denoised, -1, kernel)

        # 3. 人物区域裁剪（假设已经知道人物区域，若没有可以使用背景分割）
        # 使用简化版：裁剪图像中间区域。实际应用中，可以用深度学习背景分割进行人物区域提取
        h, w = img_sharpened.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_size = min(h, w) // 2
        img_cropped = img_sharpened[
                      center_y - crop_size: center_y + crop_size,
                      center_x - crop_size: center_x + crop_size
                      ]

        # 4. 统一调整大小
        img_resized = cv2.resize(img_cropped, target_size)

        # 5. 转换为RGB格式
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 6. 数据归一化，标准化为 [0, 1] 范围，适合预训练模型
        img_normalized = img_rgb / 255.0

        # 将预处理后的图像添加到结果列表中
        image_arrays.append(img_normalized)
    return image_list
    return image_arrays


class PersonDatabase:
    def __init__(self,cfg):
        self.extractor = CustomFeatureExtractor('osnet_x1_0',
                                                '/models/feature_extractor/weight/osnet_x1_0_imagenet.pth',
                                                'cpu')
        self.database = []
        self.cfg = cfg

    def add_person(self,person_name:str, person_images):
        """
        Add new Person to this person database.

        Parameters
        ----------
        person_name:str
            The name of the person
        person_images:list
            The base images of the person
        Returns
        -------
        person_id:int
            Assigned database id of the person
        """
        new_person = Person(person_name,base_size=self.cfg['person_database']['base_size'],recent_size=self.cfg['person_database']['recent_size'])
        person_features = self.extractor.get_normalized_result(preprocess_images(person_images))
        for i,(person_image,person_feature) in enumerate(zip(person_images,person_features)):
            ret = new_person.update_image_and_feature(person_image,person_feature,'base')
            if not ret:
                print("Update person database error")
                exit(1)
        new_person.fuse_feature()
        self.database.append(new_person)

    def update_person_feature(self, update_names: list, update_images: list, times: int, reset_names:list):
        """
        Dynamically update person's feature vector or reset(clear) it if needed and rebuild database index

        Parameters
        ----------
        update_names: list
            Names of person whose feature need to be update
        update_images: list
            New images of person who need to be update
        times: int
            Repeat the number of times the memory is reinforced
        reset_names
            Name of person whose information need to be reset(clear)
        Returns
        -------

        """
        for person_name, person_image in zip(update_names, update_images):
            person_id = -1
            for i, person in enumerate(self.database):
                if person_name == person.get_name():
                    person_id = i
                    break
            if person_id != -1:
                person_features = self.extractor.get_normalized_result(preprocess_images([person_image]))
                for i in range(times): # Reinforce the memory for several times
                    ret = self.database[person_id].update_image_and_feature(person_image, person_features[0], 'recent')
                    if not ret:
                        print("Update person database error")
                        exit(1)
                self.database[person_id].fuse_feature()
            else:
                print(f"Person '{person_name}' not found in the database.")
        person_id = -1
        for reset_name in reset_names:
            for i, person in enumerate(self.database):
                if reset_name == person.get_name():
                    person_id = i
                    break
            if person_id != -1:
                self.database[person_id].clear_recent_image_and_feature()
                self.database[person_id].fuse_feature()
                print(f"Reset feature for person '{reset_name}' with ID: {person_id}")
        # Rebuild the index

    def search(self, query_image, top_k=3):
        """
        Search for the person who has the closest L2 distance of the query image

        Parameters
        ----------
        query_image: nd.nparray

        top_k: int
            The top k close person need to be return

        """
        query_feature = self.extractor.get_normalized_result(preprocess_images(query_image))
        query_feature = query_feature.detach().cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_feature, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.names[idx], distances[0][i]))
        return results

    # def multi_frame_search(self, query_images:list, top_k=3):
    #     """
    #     Search for the person who has the closest L2 distance of the query image
    #
    #     Parameters
    #     ----------
    #     query_images: list
    #
    #     top_k: int
    #         The top k close person need to be return
    #
    #     """
    #     # Try to use more images to get fused feature, and improve the accuracy
    #     query_features = self.extractor.get_normalized_result(query_images)
    #     fused_feature = torch.mean(torch.stack(list(query_features)), dim=0)
    #     query_feature = fused_feature.detach().cpu().numpy().reshape(1, -1)
    #     distances, indices = self.index.search(query_feature, top_k)
    #     results = []
    #     for i, idx in enumerate(indices[0]):
    #         if idx != -1:
    #             results.append((self.names[idx], distances[0][i]))
    #     return results

    def calculate_similarity_and_sort(self, query_feature, top_k):
        results = []
        for person in self.database:
            name = person.get_name()
            similarity = person.calculate_cosine_similarity(query_feature)
            results.append((name, similarity))

        # 按照相似度从大到小排序，取前 top_k 个结果
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

        return sorted_results

    def multi_frame_search(self, query_images: list, top_k=3):
        # 存储每个人的累积相似度总和和计数
        person_similarity = {}

        if query_images is not None:
            query_features = self.extractor.get_normalized_result(preprocess_images(query_images))

            # 遍历所有query_feature，计算相似度并累计
            for query_feature in query_features:
                result = self.calculate_similarity_and_sort(query_feature, top_k)
                for name, similarity in result:
                    if name not in person_similarity:
                        person_similarity[name] = {'total_similarity': 0, 'count': 0}
                    person_similarity[name]['total_similarity'] += similarity
                    person_similarity[name]['count'] += 1

            # 计算每个人的平均相似度
            avg_similarity = []
            for name, data in person_similarity.items():
                avg_sim = data['total_similarity'] / data['count']
                avg_similarity.append((name, avg_sim))

            # 按照平均相似度从大到小排序，取前 top_k 个
            sorted_avg_similarity = sorted(avg_similarity, key=lambda x: x[1], reverse=True)[:top_k]

            return sorted_avg_similarity
