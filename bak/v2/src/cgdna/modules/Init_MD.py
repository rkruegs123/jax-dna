import scipy.io as sio
import copy
class init_MD_data:
    def __init__(self, nbp=None, seq=None,
                 shape=None, shape_cg=None, shape_inter=None,
                 s1b=None, stiff_me=None, nsnap=None,
                 shape_sym=None, s1b_sym=None, stiff_me_sym=None, m=None,
                 s1b_inter = None,s1b_sym_inter = None,stiff_me_inter = None,stiff_me_sym_inter = None,
                 s1b_cg = None,s1b_sym_cg = None,stiff_me_cg = None,stiff_me_sym_cg = None,stiff_mre=None,
                 stiff={}):
        self.nbp = nbp
        self.seq = seq
        self.shape = shape
        self.s1b = s1b
        self.stiff_me = stiff_me
        self.stiff_mre = stiff_mre
        self.shape_sym = shape_sym
        self.s1b_sym = s1b_sym
        self.stiff_me_sym = stiff_me_sym
        self.nsnap = nsnap
        self.m = m

    def load_data(self,load_name):
        data = sio.loadmat(load_name,squeeze_me=True)
        self.seq = data['olig']['seq']
        self.shape = data['olig']['shape']
        self.s1b = data['olig']['s1b']
        self.stiff_me = data['olig']['stiff_me']
        try:
            self.shape_sym = data['olig']['shape_sym']
            self.s1b_sym = data['olig']['s1b_sym']
            self.stiff_me_sym = data['olig']['stiff_me_sym']
        except:
            self.shape_sym = None
            self.s1b_sym = None
            self.stiff_me_sym = None


        try:
            self.shape_inter = data['olig']['shape_inter']
            self.shape_sym_inter = data['olig']['shape_sym_inter']
            self.s1b_inter = data['olig']['s1b_inter']
            self.s1b_sym_inter = data['olig']['s1b_sym_inter']
            self.stiff_me_inter = data['olig']['stiff_me_inter']
            self.stiff_me_sym_inter = data['olig']['stiff_me_sym_inter']
        except:
            self.s1b_inter = None
            self.s1b_sym_inter = None
            self.stiff_me_inter = None
            self.stiff_me_sym_inter = None
            self.shape_inter = None
            self.shape_sym_inter = None


        try:
            self.shape_cg = data['olig']['shape_cg']
            self.shape_sym_cg = data['olig']['shape_sym_cg']
            self.s1b_cg = data['olig']['s1b_cg']
            self.s1b_sym_cg = data['olig']['s1b_sym_cg']
            self.stiff_me_cg = data['olig']['stiff_me_cg']
            self.stiff_me_sym_cg = data['olig']['stiff_me_sym_cg']
        except:
            self.s1b_cg = None
            self.s1b_sym_cg = None
            self.stiff_me_cg = None
            self.stiff_me_sym_cg = None
            self.shape_cg = None
            self.shape_sym_cg = None

        try:
            self.stiff_mre = data['olig']['stiff_mre']
        except:
            self.stiff_mre = None


        try:
            self.m = data['olig']['m']
        except:
            self.m = None
            
        self.nsnap = data['olig']['nsnap']
        self.nbp = data['olig']['nbp']
        return self

    def choose_seq(self, seq_list):
        sub_data = copy.deepcopy(self)
        sub_data.nbp = [self.nbp[i] for i in seq_list]
        sub_data.seq = [self.seq[i] for i in seq_list]
        sub_data.shape = [self.shape[i] for i in seq_list]
        sub_data.s1b = [self.s1b[i] for i in seq_list]
        sub_data.stiff_me = [self.stiff_me[i] for i in seq_list]
        try:
            sub_data.shape_sym = [self.shape_sym[i] for i in seq_list]
            sub_data.s1b_sym = [self.s1b_sym[i] for i in seq_list]
            sub_data.stiff_me_sym = [self.stiff_me_sym[i] for i in seq_list]
        except:
            sub_data.shape_sym = None
            sub_data.s1b_sym   = None
            sub_data.stiff_me_sym = None

        try:
            sub_data.s1b_inter = [self.s1b_inter[i] for i in seq_list]
            sub_data.s1b_sym_inter = [self.s1b_sym_inter[i] for i in seq_list]
            sub_data.stiff_me_inter= [self.stiff_me_inter[i] for i in seq_list]
            sub_data.stiff_me_sym_inter = [self.stiff_me_sym_inter[i] for i in seq_list]
            sub_data.shape_inter = [self.shape_inter[i] for i in seq_list]
            sub_data.shape_sym_inter =  [self.shape_sym_inter[i] for i in seq_list]
        except:
            sub_data.s1b_inter = None
            sub_data.s1b_sym_inter = None
            sub_data.stiff_me_inter = None
            sub_data.stiff_me_sym_inter = None
            sub_data.shape_inter = None
            sub_data.shape_sym_inter = None


        try:
            sub_data.s1b_cg = [self.s1b_cg[i] for i in seq_list]
            sub_data.s1b_sym_cg = [self.s1b_sym_cg[i] for i in seq_list]
            sub_data.stiff_me_cg= [self.stiff_me_cg[i] for i in seq_list]
            sub_data.stiff_me_sym_cg = [self.stiff_me_sym_cg[i] for i in seq_list]
            sub_data.shape_cg = [self.shape_cg[i] for i in seq_list]
            sub_data.shape_sym_cg =  [self.shape_sym_cg[i] for i in seq_list]

        except:
            sub_data.s1b_cg = None
            sub_data.s1b_sym_cg = None
            sub_data.stiff_me_cg = None
            sub_data.stiff_me_sym_cg = None
            sub_data.shape_cg = None
            sub_data.shape_sym_cg = None


        try:
            sub_data.stiff_mre = [self.stiff_mre[i] for i in seq_list]
        except:
            sub_data.stiff_mre = None



        try:
            sub_data.m = [self.m[i] for i in seq_list]
        except:
            sub_data.m = None
            
        sub_data.nsnap = [self.nsnap[i] for i in seq_list]
        return sub_data

