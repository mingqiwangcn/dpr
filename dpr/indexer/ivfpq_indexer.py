import faiss
import logging
import os
from .faiss_indexers import DenseIndexer 
import numpy as np

logger = logging.getLogger()

class IVFPQIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 1000000):
        super(IVFPQIndexer, self).__init__(buffer_size=buffer_size)
        self.counter = 0
         
    def init_index(self, vector_sz):
        self.index = faiss.index_factory(vector_sz, "IVF256,PQ32x8", faiss.METRIC_INNER_PRODUCT)
    
    def train_needed(self):
        return True
     
    def train(self, vectors):
        logger.info("training index with sample size %d" % len(vectors))
        self.index.train(vectors)
        logger.info('training done')
   
    def index_data(self, data):
        self.index.set_direct_map_type(faiss.DirectMap.Hashtable)
        p_ids = [int(a[0].split(':')[1]) for a in data]
        emb_lst = [np.reshape(a[1], (1, -1)) for a in data]
        p_embs = np.concatenate(emb_lst, axis=0)
        self.index.add_with_ids(p_embs, p_ids)
        self.counter += len(p_ids) 
        logger.info("data indexed %d", self.counter)
    
    def index_exists(self, index_dir):
        index_file = os.path.join(index_dir, "index.dpr")
        return os.path.isfile(index_file) 
     
    def serialize(self, out_dir):
        logger.info("Serializing index to %s", out_dir)
        if not os.path.isdir(file):
            os.makedirs(out_dir)
        index_file = os.path.join(out_dir, "index.dpr")
        faiss.write_index(self.index, index_file)
    
    def deserialize(self, out_dir):
        logger.info("Loading index from %s", out_dir)
        index_file = os.path.join(out_dir, "index.dpr")
        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)
    
    def search_knn(self, query, top_n=100): 
        logger.info('Searching')
        self.index.nprobe = 128
        scores, p_ids = self.index.search(query, top_n)
        db_ids = [['wiki:' + str(a) for a in item] for item in p_ids]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

