import time, os, codecs
import warnings
import numpy as np
from math import ceil
from dtaidistance import dtw as dtw_distance
from dtaidistance.dtw_barycenter import dba
import itertools

from sklearn.exceptions import ConvergenceWarning
from LearnMethods.utils import _init_avg, _set_weights
from Utils.timeseries import to_time_series_dataset
from .utils import cdist_generic

import pycuda.driver as cuda_drv
from pycuda.compiler import SourceModule

import pyopencl as cl
import pyopencl.array as cl_array

def LB_Keogh(self,s1,s2,r):
    '''
    Calculates LB_Keough lower bound to dynamic time warping. Linear
    complexity compared to quadratic complexity of dtw.
    '''
    LB_sum=0
    for ind,i in enumerate(s1):
        
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    
    return np.sqrt(LB_sum)

class DTWCUDA:
    def __init__(self):
        cuda_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),"cudadtw.cu")
        fp = codecs.open(cuda_filename,"r","utf-8")
        cuda_Source_Str = fp.read()
        fp.close()
        mod = SourceModule(cuda_Source_Str)
        self.func_calc_dtw = mod.get_function("calc_dtw")
        self.GPU_Param = {}
        self.GPU_Param["Max_Grid_X"] = 65535
        self.GPU_Param["Max_Grid_Y"] = 65535
        self.GPU_Param["Max_Threads_pre_Block"] = 512
        self.GPU_Param["Max_share_memeory_per_block"] = 8*1024
        self.GPU_Param["COMPUTE_CAPABILITY_MAJOR"] = 1

        dev = cuda_drv.Device(0)
        self.GPU_Param["total_memory"] = dev.total_memory()
        dev_attrib = {}
        for att, value in dev.get_attributes().items():
            dev_attrib[str(att)] = value
        if "COMPUTE_CAPABILITY_MAJOR" in dev_attrib.keys():
            self.GPU_Param["COMPUTE_CAPABILITY_MAJOR"] = dev_attrib["COMPUTE_CAPABILITY_MAJOR"]
        if "MAX_GRID_DIM_X" in dev_attrib.keys():
            self.GPU_Param["Max_Grid_X"] = dev_attrib["MAX_GRID_DIM_X"]
        #GPU_Param["Max_Grid_X"] = 4096

        if "MAX_GRID_DIM_Y" in dev_attrib.keys():
            self.GPU_Param["Max_Grid_Y"] = dev_attrib["MAX_GRID_DIM_Y"]
        if "MAX_THREADS_PER_BLOCK" in dev_attrib.keys():
            self.GPU_Param["Max_Threads_pre_Block"] = dev_attrib["MAX_THREADS_PER_BLOCK"]
        if "MAX_SHARED_MEMORY_PER_BLOCK" in dev_attrib.keys():
            self.GPU_Param["Max_share_memeory_per_block"] = dev_attrib["MAX_SHARED_MEMORY_PER_BLOCK"]
    
    def _gen_Split_array(self, X, A, b):
        ret = list(range(0,X,A*b))
        left_X = X % (A*b)
        if left_X == 0:
            ret.append (X)
        else:
            if int(left_X/b) > 0:
                ret.append (ret[-1]+int(left_X/b)*b)
            left_X = X % b
            if left_X != 0:
                ret.append (X)
        return (ret)
    
    def run(self, s1, s2):
        #def _cuda_dtw_run (self, s1, s2, func_calc_dtw, GPU_Param):
        Block_Trg_Count1 = int(self.GPU_Param["Max_Threads_pre_Block"] / s2.shape[1])
        Block_Trg_Count2 = int(self.GPU_Param["Max_share_memeory_per_block"] / (3*s2.shape[1]*4))
        #Block_Trg_Count1 = int(GPU_Param["Max_Threads_pre_Block"])
        #Block_Trg_Count2 = int(GPU_Param["Max_share_memeory_per_block"])
        Block_Trg_Count = min (Block_Trg_Count1,Block_Trg_Count2)

        T0 = s2.shape[0]
        T1 = s2.shape[1]

        TrgS_Alignment =  Block_Trg_Count -T0 % Block_Trg_Count
        if TrgS_Alignment != Block_Trg_Count:
            s2 = np.concatenate ((s2, np.ones((TrgS_Alignment,T1),dtype=np.float32)))
        #print ("TrgS_Alignment",TrgS_Alignment,TrgS.shape[0])
        T0 = s2.shape[0]

        blockDim_x = T1 *Block_Trg_Count
        if self.GPU_Param["total_memory"] > 4*1024*1024*1024:
            MemoryUsing = 4*1024*1024*1024 - 150*1024*1024
        else:
            if self.GPU_Param["COMPUTE_CAPABILITY_MAJOR"] <= 2:
                if self.GPU_Param["total_memory"] > 128*1024*1024:
                    MemoryUsing = 160*1024*1024
                else:
                    MemoryUsing = 64*1024*1024
            else:
                MemoryUsing = int(self.GPU_Param["total_memory"] - 150*1024*1024)
            
        # ---
        #MemoryUsing = 512*1024*1024
        # ---
        gridDim_X = int (MemoryUsing /(Block_Trg_Count*T1*4))
        if gridDim_X > self.GPU_Param["Max_Grid_X"]:
            gridDim_X = self.GPU_Param["Max_Grid_X"]
        gridDim_Y = int (MemoryUsing /(gridDim_X *Block_Trg_Count* T1*4))
        if gridDim_Y > self.GPU_Param["Max_Grid_Y"]:
            gridDim_Y = self.GPU_Param["Max_Grid_Y"]
        if gridDim_Y == 0:
            gridDim_Y = 1
        
        Splits = list(range(0,T0,gridDim_X*gridDim_Y*Block_Trg_Count))
        if T0 %(gridDim_X*gridDim_Y*Block_Trg_Count) == 0:
            Splits.append (T0)
        else:
            t_Splits = list(range(Splits[-1], T0, gridDim_X*Block_Trg_Count))
            if len(t_Splits) > 1:
                Splits.append(Splits[-1] + (len(t_Splits)-1)*gridDim_X*Block_Trg_Count)
            if T0 %(gridDim_X*Block_Trg_Count) == 0:
                Splits.append (T0)
            else:
                t_Splits = list(range(Splits[-1], T0, Block_Trg_Count))
                if len(t_Splits) > 1:
                    Splits.append(Splits[-1] + len(t_Splits)*Block_Trg_Count)

        print ("grids,block_DimX",gridDim_X,gridDim_Y,Block_Trg_Count,blockDim_x)
        print ("Splits",Splits)
        allret = np.ones ((s1.shape[0],T0))

        for j in range(len(Splits)-1):
            TrgS_sub = s2[Splits[j]:Splits[j+1],:]
            Gy = int(TrgS_sub.shape[0]/(Block_Trg_Count *gridDim_X))
            if Gy == 0:
                Gy = 1
                Gx = int(TrgS_sub.shape[0] /Block_Trg_Count)
            else:
                Gx = int(TrgS_sub.shape[0] /(Block_Trg_Count*Gy))
            t = np.reshape(TrgS_sub,(TrgS_sub.shape[0]*TrgS_sub.shape[1]))

            print ("grid_run T0,Gx,Gy,Tbits ",TrgS_sub.shape[0],Gx,Gy,t.nbytes/1024/1024)
            #gpu_t = cuda_drv.mem_alloc(t.nbytes)
            #cuda_drv.memcpy_htod(gpu_t, t)

            share_memory_size = int(3*Block_Trg_Count*TrgS_sub.shape[1]*4)
            #print ("share_memory_size ", share_memory_size)

            r = np.ones(Block_Trg_Count*Gx*Gy,dtype=np.float32)
            gpu_r = cuda_drv.mem_alloc(r.nbytes)
            cuda_drv.memcpy_htod(gpu_r, r)

            print("Block_Trg_Count: ", Block_Trg_Count)

            for i in range(s1.shape[0]):
                s = s1[i,:]
                self.func_calc_dtw(
                    np.uint32(s1.shape[1]) ,\
                    np.uint32(TrgS_sub.shape[1]) ,\
                    np.uint32(Block_Trg_Count) ,\
                    cuda_drv.In(s), cuda_drv.In(t), \
                    gpu_r,\
                    shared = share_memory_size,\
                    block=(blockDim_x,1,1), \
                    grid=(Gx,Gy,1))
                cuda_drv.Context.synchronize()
                cuda_drv.memcpy_dtoh (r, gpu_r)
                #cuda_drv.memcpy_dtoh_async (r, gpu_r)

                # HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\GraphicsDrivers
                # TdrLevel 0
                # TdrDelay 12000000

                allret[i,Splits[j]:Splits[j+1]] = r
            """
            if i%50 == 0:
                print (i,"\t",datetime.datetime.now(),time.time()-t0)
                t0 = time.time()
            """
        print("TrgS_Alignment:", TrgS_Alignment)
        if TrgS_Alignment != 0:
            return allret[:,:-TrgS_Alignment]
        else:
            return allret

class DTWOGL:
    def __init__(self):
        opencl_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),"opencldtw.cl")
        fp = codecs.open(opencl_filename,"r","utf-8")
        opencl_Source_Str = fp.read()
        fp.close()

        self.ctx = cl.create_some_context()
        self.prg = cl.Program(self.ctx, opencl_Source_Str).build()

        self.dev_Param = {}
        self.dev_Param["MAX_MEM_ALLOC_SIZE"] = 1024*1024*64
        self.dev_Param["LOCAL_MEM_SIZE"] = 1024*8
        self.dev_Param["MAX_WORK_ITEM_SIZES"] = [512,512]
        self.dev_Param["MAX_WORK_GROUP_SIZE"] = 512
        self.dev_Param["MAX_WORK_GROUP_SIZE"] = 512
        self.dev_Param["MAX_MEM_ALLOC_SIZE"] = \
            self.ctx.devices[0].get_info(getattr(cl.device_info, "MAX_MEM_ALLOC_SIZE"))
        self.dev_Param["MAX_MEM_ALLOC_SIZE"] = \
            self.dev_Param["MAX_MEM_ALLOC_SIZE"] - 64*1024*1024
        self.dev_Param["LOCAL_MEM_SIZE"] = \
            self.ctx.devices[0].get_info(getattr(cl.device_info, "LOCAL_MEM_SIZE"))
        self.dev_Param["MAX_WORK_ITEM_SIZES"] = \
            self.ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_ITEM_SIZES"))
        self.dev_Param["MAX_WORK_GROUP_SIZE"] = \
            self.ctx.devices[0].get_info(getattr(cl.device_info, "MAX_WORK_GROUP_SIZE"))
        #print (dev_Param)
        self.queue = cl.CommandQueue(self.ctx)
    
    def run (self, s1, s2):
        #MAX_MEM_ALLOC_SIZE
        cot1 = int(self.dev_Param["LOCAL_MEM_SIZE"] / (s2.shape[1] *4 *3))
        cot2 = int(self.dev_Param["MAX_WORK_ITEM_SIZES"][0] / s2.shape[1])
        TRG_COT = min(cot1, cot2)
        Grp_Cot = int(self.dev_Param["MAX_MEM_ALLOC_SIZE"] / (s2.shape[1] *4 *TRG_COT))

        T0 = s2.shape[0]
        T1 = s2.shape[1]

        TrgS_Alignment =  TRG_COT -T0 % TRG_COT
        if TrgS_Alignment != TRG_COT:
            s2 = np.concatenate ((s2, np.ones((TrgS_Alignment,T1),dtype=np.float32)))
        T0 = s2.shape[0]
        #print ("TrgS_Alignment,TRG_COT",TrgS_Alignment,TRG_COT)

        Splits = list(range(0, T0, Grp_Cot *TRG_COT))
        Splits.append (T0)
        allret = np.empty ((s1.shape[0],s2.shape[0]), dtype=np.float32)

        for j in range(len(Splits)-1):
            TrgS_sub = s2[Splits[j]:Splits[j+1],:]
            Ts0 = TrgS_sub.shape[0]
            Ts1 = TrgS_sub.shape[1]
            local_size  = TRG_COT *Ts1
            global_size = Ts0 *Ts1
            
            t = np.reshape(TrgS_sub,(Ts0 *Ts1))
            t_dev = cl_array.to_device(self.queue, t)
            #print ("local_size, global_size ",local_size,global_size,t.nbytes/1024/1024)
    
            SRC_LEN = s1.shape[1]
            TRG_LEN = s2.shape[1]

            for i in range(s1.shape[0]):
                s = s1[i,:]
                s_dev = cl_array.to_device(self.queue, s)
                r_dev = cl_array.empty (self.queue, (Ts0,), dtype=np.float32)
                shared_mem_size = Ts1 *TRG_COT *4

                self.prg.opencl_dtw(self.queue, (global_size,), (local_size,), \
                    np.uint32(SRC_LEN),np.uint32(TRG_LEN),np.uint32(TRG_COT),
                    s_dev.data, t_dev.data, r_dev.data,\
                    cl.LocalMemory(shared_mem_size),
                    cl.LocalMemory(shared_mem_size),
                    cl.LocalMemory(shared_mem_size)
                    )
                r = r_dev.get()
                allret[i,Splits[j]:Splits[j+1]] = r
                #print(la.norm((dest_dev - (a_dev+b_dev)).get()))

        if TrgS_Alignment != TRG_COT:
            allret = allret[:,0:-TrgS_Alignment]
        return (allret)
    
class DTW:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        #self.exec_type = exec_type
        #self.options = options
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def use_cache(self):
        if not hasattr(self,"cache"):
            return False
        else:
            return self.cache
         
    def distance(self, dataset1, dataset2, inv = False):
        if dataset2 is None:
            dataset2 = np.copy(dataset1)
            
        if inv:
            dataset1 = np.array(dataset1).T
            dataset2 = np.array(dataset2).T
            
        n1 = dataset1.shape[0]
        n2 = dataset2.shape[0]
        dist = np.empty((n1, n2), dtype=np.float64)
        
        for i,j in list(itertools.product(np.arange(0,n1),np.arange(0,n2))):
            dist[i][j] = self._distance(np.array(dataset1[i],dtype=float), np.array(dataset2[j],dtype=float))
            if np.isinf(dist[i][j]):
                print("dtw.distance.dataset1[i]:", dataset1[i])
                print("dtw.distance.dataset2[j]:", dataset2[j])
        
        return dist
    
    #@jit(nopython=True, cache=True)
    def _distance(self, serie1, serie2):
        serie1 = serie1[~np.isnan(serie1)]
        serie2 = serie2[~np.isnan(serie2)]
        w = int(min(len(serie1), len(serie2)) * self.options.dtw_sakoechiba_w)
        if self.use_cache:
            found, dist = self.cache_data.search(serie1=serie1.tolist(), serie2=serie2.tolist())
            if not found:
                if self.exec_type == 'cpu':
                    dist = dtw_distance.distance_fast(s1=serie1,s2=serie2,window=w,use_pruning=True)
                    self.cache_data.append(serie1=serie1.tolist(), serie2=serie2.tolist(), distance=dist)
                    return dist
                elif self.exec_type == 'gpu':
                    dist = self.cuda_dtw(s1=serie1,s2=serie2)
                    return dist
                elif self.exec_type == 'opengl':
                    dist = self.opencl_dtw(s1=serie1,s2=serie2)
                    return dist
            else:
                return dist
        else:
            if self.exec_type == 'cpu':
                dist = dtw_distance.distance_fast(s1=serie1,s2=serie2,window=w,use_pruning=True)
                return dist
            elif self.exec_type == 'gpu':
                dist = self.cuda_dtw(s1=serie1,s2=serie2)
                return dist
            elif self.exec_type == 'opengl':
                dist = self.opencl_dtw(s1=serie1,s2=serie2)
                return dist
        
    def cuda_dtw (self, s1, s2, show_time = False):
        t0 = time.time()
        dtw_cuda = DTWCUDA()
        ret = dtw_cuda.run(s1,s2)
        if show_time:
            print ("cuda run time:  ",time.time()-t0)
        return (ret)
    
    def opencl_dtw (self, s1, s2, show_time=False):
        t0 = time.time()
        dtw_opencl = DTWOGL()
        ret = dtw_opencl.run (s1, s2)
        if show_time:
            print ("opencl run time:  ",time.time()-t0)
        return (ret)
    
    def _check_sakoe_chiba_params(self, n_timestamps_1, n_timestamps_2):
            """Check and set some parameters of the sakoe-chiba band."""
            if not isinstance(n_timestamps_1, (int, np.integer)):
                raise TypeError("'n_timestamps_1' must be an integer.")
            else:
                if not n_timestamps_1 >= 2:
                    raise ValueError("'n_timestamps_1' must be an integer greater than"
                                    " or equal to 2.")
            window_size = self.options.dtw_sakoechiba_w
            if not isinstance(window_size, (int, np.integer, float, np.floating)):
                raise TypeError("'window_size' must be an integer or a float.")
            n_timestamps = max(n_timestamps_1, n_timestamps_2)

            if isinstance(window_size, (float, np.floating)):
                if not 0. <= window_size <= 1.:
                    raise ValueError("The given 'window_size' is a float, "
                                    "it must be between "
                                    "0. and 1. To set the size of the sakoe-chiba "
                                    "manually, 'window_size' must be an integer.")
                window_size = ceil(window_size * (n_timestamps - 1))
            else:
                if not 0 <= window_size <= (n_timestamps - 1):
                    raise ValueError(
                        "The given 'window_size' is an integer, it must "
                        "be greater "
                        "than or equal to 0 and lower than max('n_timestamps_1', "
                        "'n_timestamps_2')."
                    )
            
            scale = (n_timestamps_2 - 1) / (n_timestamps_1 - 1)

            if n_timestamps_2 > n_timestamps_1:
                window_size = max(window_size, scale / 2)
                horizontal_shift = 0
                vertical_shift = window_size
            elif n_timestamps_1 > n_timestamps_2:
                window_size = max(window_size, 0.5 / scale)
                horizontal_shift = window_size
                vertical_shift = 0
            else:
                horizontal_shift = 0
                vertical_shift = window_size
            return scale, horizontal_shift, vertical_shift
        
    def path(self, s1, s2):
        return dtw_distance.warping_path_fast(from_s=s1, to_s=s2)
    
    def _sakoe_chiba_band(self, n_timestamps_1, n_timestamps_2=None):
        """Compute the Sakoe-Chiba band.

        Parameters
        ----------
        n_timestamps_1 : int
            The size of the first time series.

        n_timestamps_2 : int (optional, default None)
            The size of the second time series. If None, set to `n_timestamps_1`.


        Returns
        -------
        region : array, shape = (2, n_timestamps_1)
            Constraint region. The first row consists of the starting indices
            (included) and the second row consists of the ending indices (excluded)
            of the valid rows for each column.

        References
        ----------
        .. [1] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization
            for spoken word recognition”. IEEE Transactions on Acoustics,
            Speech, and Signal Processing, 26(1), 43-49 (1978).

        Examples
        --------
        >>> from pyts.metrics import sakoe_chiba_band
        >>> print(sakoe_chiba_band(5, window_size=0.5))
        [[0 0 0 1 2]
        [3 4 5 5 5]]

        """
        if n_timestamps_2 is None:
            n_timestamps_2 = n_timestamps_1
        scale, horizontal_shift, vertical_shift = self._check_sakoe_chiba_params(n_timestamps_1, n_timestamps_2)

        lower_bound = scale * (np.arange(n_timestamps_1) - horizontal_shift) - vertical_shift
        lower_bound = np.round(lower_bound, 2)
        lower_bound = np.ceil(lower_bound)
        upper_bound = scale * (np.arange(n_timestamps_1) + horizontal_shift) + vertical_shift
        upper_bound = np.round(upper_bound, 2)
        upper_bound = np.floor(upper_bound) + 1
        region = np.asarray([lower_bound, upper_bound]).astype('int64')
        region = (region, n_timestamps_1, n_timestamps_2)
        return region
    
    def cdist(self, dataset1, dataset2, n_jobs=None, verbose=0):
        return cdist_generic(dist_fun=self.distance, dataset1=dataset1, dataset2=dataset2,
                        n_jobs=n_jobs, verbose=verbose,
                        compute_diagonal=False)
    
    def barycenter_averaging(self, X, init_barycenter=None):
        w = int(min(len(X), len(init_barycenter)) * self.options.dtw_sakoechiba_w)
        print("dtw.barycenter_aveeraging.init_barycenter:", np.array(list(init_barycenter.ravel())))
        return dba(s=X.ravel(), c=np.array(list(init_barycenter.ravel())), use_c=True, window=w, use_pruning=True)

    """ @jit(nopython=True, cache=True)
    def dtw_cpu(self, x, y):
        len_x, len_y = len(x), len(y)
        
        len_x, len_y = len(self.x), len(self.y)
        if self.window is None:
            self.window = [(i, j) for i in xrange(len_x) for j in xrange(len_y)]
        self.window = [(i + 1, j + 1) for i, j in self.window]
        D = defaultdict(lambda: [float('inf')])
        D[0, 0] = [0, 0, 0]
        for i, j in self.window:
            D[i, j] = min([D[i-1, j][0], i-1, j], [D[i, j-1][0], i, j-1], [D[i-1, j-1][0], i-1, j-1], key=lambda a: a[0])
            D[i, j][0] += self.dist(self.x[i-1], self.y[j-1])
        path = []
        i, j = len_x, len_y
        while not (i == j == 0):
            path.append((i-1, j-1))
            i, j = D[i, j][1], D[i, j][2]
        path.reverse()
        return (D[len_x, len_y][0], path) """
        
""" Falta probar
        from numba import jit
        from scipy import spatial

        @jit
        def D_from_cost(cost, D):
        # operates on D inplace
        ns, nt = cost.shape
        for i in range(ns):
            for j in range(nt):
            D[i+1, j+1] = cost[i,j]+min(D[i, j+1], D[i+1, j], D[i, j])
            # avoiding the list creation inside mean enables better jit performance
            # D[i+1, j+1] = cost[i,j]+min([D[i, j+1], D[i+1, j], D[i, j]])

        @jit
        def get_d(D, matchidx):
        ns = D.shape[0] - 1
        nt = D.shape[1] - 1
        d = D[ns,nt]

        matchidx[0,0] = ns - 1
        matchidx[0,1] = nt - 1
        i = ns
        j = nt
        for k in range(1, ns+nt+3):
            idx = 0
            if not (D[i-1,j] <= D[i,j-1] and D[i-1,j] <= D[i-1,j-1]):
            if D[i,j-1] <= D[i-1,j-1]:
                idx = 1
            else:
                idx = 2

            if idx == 0 and i > 1 and j > 0:
            # matchidx.append([i-2, j-1])
            matchidx[k,0] = i - 2
            matchidx[k,1] = j - 1
            i -= 1
            elif idx == 1 and i > 0 and j > 1:
            # matchidx.append([i-1, j-2])
            matchidx[k,0] = i-1
            matchidx[k,1] = j-2
            j -= 1
            elif idx == 2 and i > 1 and j > 1:
            # matchidx.append([i-2, j-2])
            matchidx[k,0] = i-2
            matchidx[k,1] = j-2
            i -= 1
            j -= 1
            else:
            break

        return d, matchidx[:k]


        def seqdist2(seq1, seq2):
            ns = len(seq1)
            nt = len(seq2)

            cost = spatial.distance_matrix(seq1, seq2)

            # initialize and update D
            D = np.full((ns+1, nt+1), np.inf)
            D[0, 0] = 0
            D_from_cost(cost, D)

            matchidx = np.zeros((ns+nt+2,2), dtype=np.int)
            d, matchidx = get_d(D, matchidx)
            return d, matchidx[::-1].tolist()
        
        seq1 = np.random.randint(100, size=(100, 2)) #Two dim sequences
        seq2 = np.random.randint(100, size=(100, 2))
        
        %timeit seqdist2(seq1, seq2) # 1000 loops, best of 3: 365 µs per loop
        """
    
    