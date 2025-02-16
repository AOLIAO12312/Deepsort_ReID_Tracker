import faiss
import numpy as np
from src.person import Person
from models.feature_extractor.custom_feature_extractor import CustomFeatureExtractor


class PersonDatabase:
    def __init__(self):
        self.extractor = CustomFeatureExtractor('osnet_x1_0',
                                                '/Volumes/Disk_1/ApplicationData/PythonProject/ReID-Tracker/models/feature_extractor/osnet_x1_0_imagenet.pth',
                                                'cpu')
        self.database = []
        self.index = None
        self.features = []
        self.names = []

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
        new_person = Person(person_name,base_size=5,recent_size=20)
        person_features = self.extractor.get_normalized_result(person_images)
        for i,(person_image,person_feature) in enumerate(zip(person_images,person_features)):
            ret = new_person.update_image_and_feature(person_image,person_feature,'base')
            if not ret:
                print("Update person database error")
                exit(1)
        new_person.fuse_feature()
        self.database.append(new_person)
        if new_person.get_fused_feature() is not None:
            feature = new_person.get_fused_feature().detach().cpu().numpy().reshape(1, -1)
            person_id = len(self.database) - 1
            if self.index is None:
                index = faiss.IndexFlatL2(feature.shape[1])
                self.index = faiss.IndexIDMap(index)
                self.index.add_with_ids(feature, np.array([len(self.database) - 1], dtype=np.int64))
            else:
                self.index.add_with_ids(feature, np.array([len(self.database) - 1], dtype=np.int64))

            self.features.append(feature)
            self.names.append(new_person.get_name())
            return person_id

    def update_person_feature_and_rebuild_index(self, update_names: list, update_images: list, times: int, reset_names:list):
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
        for person_name, person_images in zip(update_names, update_images):
            person_id = -1
            for i, name in enumerate(self.names):
                if person_name == name:
                    person_id = i
                    break
            if person_id != -1:
                person_features = self.extractor.get_normalized_result(person_images)
                for i in range(times): # Reinforce the memory for several times
                    for j, (person_image, person_feature) in enumerate(zip(person_images, person_features)):
                        ret = self.database[person_id].update_image_and_feature(person_image, person_feature, 'recent')
                        if not ret:
                            print("Update person database error")
                            exit(1)
                self.database[person_id].fuse_feature()
                new_feature = self.database[person_id].get_fused_feature()
                new_feature = new_feature.detach().cpu().numpy().reshape(-1)
                self.features[person_id] = new_feature
                # print(f"Updated feature for person '{self.names[person_id]}' with ID: {person_id}")
            else:
                print(f"Person '{person_name}' not found in the database.")
        person_id = -1
        for reset_name in reset_names:
            for i, name in enumerate(self.names):
                if reset_name == name:
                    person_id = i
                    break
            if person_id != -1:
                self.database[person_id].clear_recent_image_and_feature()
                self.database[person_id].fuse_feature()
                new_feature = self.database[person_id].get_fused_feature()
                new_feature = new_feature.detach().cpu().numpy().reshape(-1)
                self.features[person_id] = new_feature
                print(f"Reset feature for person '{self.names[person_id]}' with ID: {person_id}")
        # Rebuild the index
        features_array = np.vstack(self.features)  # ensure the features if 2-dimensional
        d = features_array.shape[1]
        new_index = faiss.IndexFlatL2(d)
        new_index = faiss.IndexIDMap(new_index)
        new_index.add_with_ids(features_array, np.array(range(len(self.features)), dtype=np.int64))
        self.index = new_index

    def search(self, query_image, top_k=3):
        """
        Search for the person who has the closest L2 distance of the query image

        Parameters
        ----------
        query_image: nd.nparray

        top_k: int
            The top k close person need to be return

        """
        query_feature = self.extractor.get_normalized_result(query_image)
        query_feature = query_feature.detach().cpu().numpy().reshape(1, -1)
        distances, indices = self.index.search(query_feature, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.names[idx], distances[0][i]))
        return results
