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
        self.fac_str = None
    
    def factory_string_needed(self):
        return True
        
    def set_factory_string(self, fac_str):
        self.fac_str = fac_str
     
    def init_index(self, vector_sz):
        #example factory string, "OPQ64_768,IVF65536,PQ64"
        assert self.fac_str is not None, "Factory string must be specified"
        self.index = faiss.index_factory(vector_sz, self.fac_str, faiss.METRIC_INNER_PRODUCT)
    
    def train_needed(self):
        return True
     
    def train(self, vectors):
        logger.info("training index with sample size %d" % len(vectors))
        index_ivf = faiss.extract_index_ivf(self.index)
        logger.info('d=%d' % index_ivf.d)
        cls_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
        index_ivf.clustering_index = cls_index
        self.index.train(vectors)
        logger.info('training done')
   
    def index_data(self, data):
        index_ivf = faiss.extract_index_ivf(self.index)
        index_ivf.set_direct_map_type(faiss.DirectMap.Hashtable)
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
        obj = faiss.extract_index_ivf(self.index)
        obj.nprobe = 128
        scores, p_ids = self.index.search(query, top_n)
        db_ids = [['wiki:' + str(a) for a in item] for item in p_ids]
        result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
        return result

